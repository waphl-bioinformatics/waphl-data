#!/usr/bin/env python3
import os
import re
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, DefaultDict

import streamlit as st
import pandas as pd
import numpy as np
import fsspec
from collections import defaultdict

from shared import data_processing, ui
from dashboard.inspect import workflow_specific_functions as wsf

# ---------- GLOBALS ----------
BASE_COLS = ["accept", "reject", "id_alt", "run", "workflow_alt"]

# File types that belong to exactly one sample. These must never be filled from
# run-level (id_alt == "") files: doing so hands every sample that lacks its own
# file ALL of the run's unassigned files, i.e. other samples' assemblies.
SAMPLE_SCOPED_TYPES = {"assembly", "raw_reads"}

def apply_qc_row(row: pd.Series, criteria: List[Dict[str, Any]]) -> bool:
    """Return True if row passes all numeric min/max rules."""

    for rule in criteria or []:
        col = rule.get("column", '').lower().strip()

        if not col or col not in row:
            ui.push_message(f"apply_qc_row: Row {row.name}: {col!r} not in data", type="warning")
            return False

        val = row[col]
        if pd.isna(val) or val is None:
            return False

        if rule.get("equals") is not None and str(val) != str(rule['equals']):
            return False

        if rule.get("min_value") is not None:
            try:
                val = pd.to_numeric(val)
                min_value = float(rule["min_value"])
            except Exception:
                return False
            if val < min_value:
                return False

        if rule.get("max_value") is not None:
            try:
                val = pd.to_numeric(val)
                max_value = float(rule["max_value"])
            except Exception:
                return False
            if val > max_value:
                return False

    return True

def build_rows() -> pd.DataFrame:
    """
    Pivot the classified frame into one row per (id, id_alt, run), with one column per
    reportable file type.

    Starts from df_processed (produced by classify_files). Every sample (id, id_alt,
    run) present in df_processed gets a row, even if none of its files are
    reportable — such samples simply end up with NaN in every file-type column.
    For samples with at least one reportable file, each file type is filled from the
    sample's own reportable files, falling back to the run's global reportable files
    when a type is absent at the sample level. Missing types (after fallback) raise
    a warning.

    Returns a DataFrame: id, id_alt, run, <file types...>.
    """
    df = st.session_state.get("df_processed", pd.DataFrame())
    if df.empty:
        return

    file_types = list(st.session_state.get("inspect_reportable_file_types", []))
    if not file_types:
        ui.push_message("No reportable file types found in scheme", type="warning")
        return
    out_cols = ["id", "id_alt", "run"] + file_types

    # ---------- pivot to one row per id x run ----------
    kept = df[df["reportable"]].copy()  # reportable files only, used to fill values
    sample_kept = kept[kept["id_alt"] != ""]
    global_rows = kept[kept["id_alt"] == ""]

    # Global fallback: (run, type) -> [current, ...]  (dedup, first-seen order)
    global_lookup = defaultdict(list)
    for _, r in global_rows.iterrows():
        c = r["current"]
        if c and c not in global_lookup[(r["run"], r["type"])]:
            global_lookup[(r["run"], r["type"])].append(c)
    global_lookup = dict(global_lookup)

    sample_type_lookup = {}
    for (sid, id_alt, run), sub in sample_kept.groupby(["id", "id_alt", "run"], dropna=False):
        per_type = defaultdict(list)
        for t, c in zip(sub["type"], sub["current"]):
            if c and c not in per_type[t]:
                per_type[t].append(c)
        sample_type_lookup[(sid, id_alt, run)] = dict(per_type)

    # Every sample present in df_processed gets a row, even if it has zero
    # reportable files (in which case it just gets NaN for every file type).
    all_samples = df[df["id_alt"] != ""][["id", "id_alt", "run"]].drop_duplicates()

    missing_by_sample = {}
    final_rows = []
    for _, srow in all_samples.iterrows():
        sid, id_alt, run = srow["id"], srow["id_alt"], srow["run"]
        type_to_file = sample_type_lookup.get((sid, id_alt, run), {})
        row = {"id": sid, "id_alt": id_alt, "run": run}
        missing = []
        for ft in file_types:
            files = type_to_file.get(ft)
            if not files and ft not in SAMPLE_SCOPED_TYPES:
                files = global_lookup.get((run, ft))
            files = files or []
            if not files:
                missing.append(ft)
                row[ft] = np.nan
            elif ft == "summary":
                # Exactly one summary per sample by definition; keep it scalar
                # so the id/summary merge downstream stays a clean 1:1 join.
                if len(files) > 1:
                    ui.push_message(
                        f"Sample {sid!r} (run {run!r}) has {len(files)} summary "
                        f"files; expected one. Using first: {files[0]}",
                        type="warning",
                    )
                row[ft] = files[0]
            else:
                row[ft] = list(files)   # one or more paths
        if missing:
            missing_by_sample[(sid, id_alt, run)] = missing
        final_rows.append(row)

    df_grouped = pd.DataFrame(final_rows, columns=out_cols)

    # Workflow-specific exemptions: file types a sample legitimately won't have.
    if st.session_state.get("inspect_workflow") == "vaper":
        missing_by_sample = wsf.vaper_drop_unexpected_missing(missing_by_sample, df_grouped)

    warnings = [
        f"{sid!r}: {', '.join(missing)}"
        for (sid, _, _), missing in missing_by_sample.items()
        if missing
    ]
    if warnings:
        ui.push_message(
            "Some samples are missing reportable file types:\n\n" + "\n\n".join(warnings),
            type="warning",
        )

    # Per-sample files the gather step couldn't assign to a sample. They are no
    # longer spread across samples, so surface them instead of dropping silently.
    orphans = [
        f"{run!r}: {f}"
        for (run, ft), files in global_lookup.items()
        if ft in SAMPLE_SCOPED_TYPES
        for f in files
    ]
    if orphans:
        ui.push_message(
            "These per-sample files have no sample id and were not attached to any row:\n\n"
            + "\n\n".join(orphans),
            type="warning",
        )

    st.session_state["df_grouped"] = df_grouped

def add_summary_columns():
    """Add summary columns to the grouped frame."""
    df = st.session_state.get("df_grouped", pd.DataFrame())
    if df.empty:
        return

    summary_columns = st.session_state.get("inspect_summary_cols", {})
    if not summary_columns:
        return

    file_types = list(st.session_state.get("inspect_reportable_file_types", []))
    if not file_types:
        return

    if "summary" not in df.columns:
        return

    workflow = st.session_state.get("inspect_workflow")

    # ---- build column mapping (old -> new) ----
    col_map = {}
    for new_col, old_cols in summary_columns.items():
        if isinstance(old_cols, str):
            col_map[old_cols] = new_col
        else:
            for c in old_cols:
                col_map[c] = new_col

    # ---- load summary tables ----
    summary_dfs = []
    for uri in df["summary"].dropna().unique():
        tmp = data_processing.read_table(uri)
        tmp.columns = [str(c).lower() for c in tmp.columns]
        tmp = tmp.rename(columns=col_map)
        tmp["summary"] = uri
        summary_dfs.append(tmp)

    df_summary = (
        pd.concat(summary_dfs, ignore_index=True)
        if summary_dfs
        else pd.DataFrame()
    )

    if df_summary.empty:
        return

    # Strip a trailing "_T<number>" suffix from the summary's id column
    # (e.g. "SAMPLE123_T1" -> "SAMPLE123") so it matches df_grouped's id.
    if "id" in df_summary.columns:
        df_summary["_id_raw"] = df_summary["id"].astype(str)
        df_summary["id"] = (
            df_summary["id"]
            .astype(str)
            .str.replace(r"_T\d+$", "", regex=True)
        )

    if workflow == "vaper":
        missing_keys = [k for k in ("id", "reference") if k not in df_summary.columns]
        if missing_keys:
            ui.push_message(
                f"VAPER summary is missing column(s): {', '.join(missing_keys)}. "
                f"Columns found: {', '.join(map(str, df_summary.columns))}. Skipping summary merge.",
                type="warning",
            )
            return
        df_merged = wsf.merge_vaper(df, df_summary)
    else:
        merge_keys = ["id", "summary"]
        missing_keys = [k for k in merge_keys if k not in df_summary.columns]
        if missing_keys:
            ui.push_message(
                f"Summary tables are missing merge column(s): {', '.join(missing_keys)}. "
                f"Columns found: {', '.join(map(str, df_summary.columns))}. "
                "Check the scheme's summary_columns mapping. Skipping summary merge.",
                type="warning",
            )
            return
        df_merged = df.merge(
            df_summary.drop(columns="_id_raw", errors="ignore"),
            on=merge_keys,
            how="left",
            suffixes=("", "_summary"),
        )

    # ---- ensure BASE_COLS fields exist before selecting output columns ----
    # workflow_alt: prefer a per-sample value carried on df_processed (joined on
    # id/id_alt/run); fall back to a single workflow-level value from session state
    # if no per-sample column is available.
    df = st.session_state.get("df_processed", pd.DataFrame())
    if "workflow_alt" in df.columns:
        walt_lookup = (
            df[["id", "id_alt", "run", "workflow_alt"]]
            .dropna(subset=["workflow_alt"])
            .drop_duplicates(subset=["id", "id_alt", "run"], keep="last")
            .set_index(["id", "id_alt", "run"])["workflow_alt"]
        )
        df_merged["workflow_alt"] = (
            df_merged.set_index(["id", "id_alt", "run"]).index.map(walt_lookup)
        )
    else:
        df_merged["workflow_alt"] = st.session_state.get("inspect_workflow_alt")

    # accept/reject are computed later in apply_qc(); add placeholders now so the
    # BASE_COLS-based selection below doesn't KeyError before QC has run.
    if "accept" not in df_merged.columns:
        df_merged["accept"] = pd.NA
    if "reject" not in df_merged.columns:
        df_merged["reject"] = pd.NA

    extra_cols = ["reference"] if workflow == "vaper" else []
    keep_cols = ["id"] + BASE_COLS + list(summary_columns.keys()) + extra_cols + file_types
    seen = set()
    keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]

    for c in keep_cols:
        if c not in df_merged.columns:
            df_merged[c] = pd.NA
    df_merged = df_merged[keep_cols]

    st.session_state["df_results"] = df_merged
    

def apply_qc() -> pd.DataFrame:
    df = st.session_state.get("df_results", pd.DataFrame()).copy()
    if df.empty:
        return  
    qc_criteria = st.session_state.get("inspect_qc_criteria", [])
    if not qc_criteria:
        return

    df["accept"] = [apply_qc_row(r, qc_criteria) for _, r in df.iterrows()]
    df["reject"] = ~df["accept"]

    st.session_state["df_results"] = df


def order_df() -> pd.DataFrame:
    df = st.session_state.get("df_results", pd.DataFrame()).copy()
    if df.empty:
        return

    summary_cols_dict = st.session_state.get("inspect_summary_cols", {})
    summary_cols = list(summary_cols_dict.keys())
    present_base = [c for c in BASE_COLS if c in df.columns]
    other_cols = [c for c in df.columns if c not in present_base + summary_cols]
    df = df[present_base + summary_cols + other_cols]

    # ---- order rows by id_alt, run ----
    sort_cols = [c for c in ("run", "id_alt") if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, kind="stable", na_position="last")

    st.session_state["df_inspect"] = df


# ---------- MAIN ----------
def main():
    workflow = st.session_state.get("inspect_workflow")
    df_processed = st.session_state.get("df_processed", pd.DataFrame())

    if workflow is None or df_processed.empty:
        return

    build_rows()
    add_summary_columns()
    apply_qc()
    order_df()

if __name__ == "__main__":
    print("This module exposes main(df, workflow) → DataFrame")