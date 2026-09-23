#!/usr/bin/env python3
import re
import pandas as pd
import numpy as np

from shared import data_processing, ui

# ---------- VAPER ----------
# VAPER names consensus assemblies "<meta.id>_<ref_id>.fa.gz", where ref_id is
# already passed through safe_filename() and routinely contains underscores
# (spaces -> "_", duplicate names -> "<name>_<n>"). The reference therefore
# can't be recovered by splitting the filename; instead we rebuild the expected
# stem from each summary row (id + "_" + reference) and join on that.
VAPER_NO_REF = "no reference"
_UNSAFE_RX = re.compile(r"[^A-Za-z0-9._-]+")
_TIDY_RX = re.compile(r"\.tidy$", re.IGNORECASE)
_REPLICATE_RX = re.compile(r"_T\d+$")


def _safe(s: str) -> str:
    """Mirror of VAPER's vaper_utils.safe_filename()."""
    return _UNSAFE_RX.sub("_", str(s))


def _assembly_stem(path) -> str:
    if not isinstance(path, str) or not path:
        return np.nan
    stem = data_processing.extract_stem(path)
    return _safe(_TIDY_RX.sub("", stem)) if stem else np.nan


def explode_assemblies(df: pd.DataFrame) -> pd.DataFrame:
    """One row per assembly path; samples with no assembly keep a single NaN row."""
    df = df.copy()
    if "assembly" not in df.columns:
        df["assembly"] = np.nan

    def _as_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        return [] if pd.isna(x) else [x]

    df["assembly"] = df["assembly"].apply(_as_list)
    return df.explode("assembly", ignore_index=True)


def vaper_no_reference_ids(summary_uris) -> set:
    """
    Sample ids whose VAPER summary rows are all "no reference", i.e. VAPER found
    nothing to assemble against, so no assembly file is expected. Both the raw id
    and the id with any "_T<n>" suffix removed are returned, so either form matches.
    """
    frames = []
    for uri in pd.Series(list(summary_uris)).dropna().unique():
        tmp = data_processing.read_table(uri)
        if tmp.empty:
            continue
        tmp.columns = [str(c).lower() for c in tmp.columns]
        if {"id", "reference"} <= set(tmp.columns):
            frames.append(tmp[["id", "reference"]])
    if not frames:
        return set()

    summ = pd.concat(frames, ignore_index=True)
    summ["id"] = summ["id"].astype(str)
    summ["_no_ref"] = (
        summ["reference"].fillna("").astype(str).str.strip().str.lower() == VAPER_NO_REF
    )
    all_no_ref = summ.groupby("id")["_no_ref"].all()
    ids = set(all_no_ref[all_no_ref].index)
    return ids | {_REPLICATE_RX.sub("", i) for i in ids}


def vaper_drop_unexpected_missing(missing_by_sample: dict, df_grouped: pd.DataFrame) -> dict:
    """
    Remove "assembly" from the missing-file list of samples that VAPER reported
    as "no reference". Those samples are not expected to have an assembly.
    Samples with no summary row at all are left alone, so a truly missing
    assembly is still reported.

    missing_by_sample: {(id, id_alt, run): [missing file types]}
    """
    if "summary" not in df_grouped.columns:
        return missing_by_sample
    no_ref = vaper_no_reference_ids(df_grouped["summary"])
    if not no_ref:
        return missing_by_sample

    out = {}
    for key, missing in missing_by_sample.items():
        if str(key[0]) in no_ref:
            missing = [ft for ft in missing if ft != "assembly"]
        if missing:
            out[key] = missing
    return out


def merge_vaper(df: pd.DataFrame, df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Attach VAPER-summary.csv rows to the grouped frame.

    Rows with an assembly are matched to the summary row whose
    safe(id + "_" + reference) equals the assembly's stem (".tidy" dropped).
    Rows with no assembly fall back to the sample's "no reference" row.

    Expects df_summary to carry "_id_raw" (the id before any suffix stripping).
    """
    df = explode_assemblies(df)
    df["_row"] = range(len(df))
    df["_stem"] = df["assembly"].map(_assembly_stem)

    summ = df_summary.copy()
    summ["reference"] = summ["reference"].fillna("").astype(str)
    summ["_stem"] = (summ["_id_raw"].astype(str) + "_" + summ["reference"]).map(_safe)
    summ = summ.rename(columns={"id": "_summary_id"})

    # ---- rows with an assembly: join on the rebuilt filename stem ----
    has_asm = df["_stem"].notna()
    by_stem = summ[summ["reference"] != VAPER_NO_REF].drop_duplicates(["summary", "_stem"])
    part_asm = df[has_asm].merge(
        by_stem, on=["summary", "_stem"], how="left", suffixes=("", "_summary")
    )

    # ---- rows without an assembly: join on id -> "no reference" row ----
    no_ref = summ[summ["reference"] == VAPER_NO_REF]
    keyed = pd.concat(
        [no_ref.assign(_id_key=no_ref["_id_raw"].astype(str)),
         no_ref.assign(_id_key=no_ref["_summary_id"].astype(str))],
        ignore_index=True,
    ).drop_duplicates(["summary", "_id_key"])
    part_none = (
        df[~has_asm]
        .assign(_id_key=lambda d: d["id"].astype(str))
        .merge(keyed.drop(columns="_stem"), on=["summary", "_id_key"],
               how="left", suffixes=("", "_summary"))
        .drop(columns="_id_key")
    )

    out = pd.concat([part_asm, part_none], ignore_index=True).sort_values("_row")

    # ---- sanity checks ----
    unmatched = out[out["assembly"].notna() & out["_summary_id"].isna()]
    if not unmatched.empty:
        ui.push_message(
            "VAPER: no summary row matched these assemblies (summary columns left empty):\n\n"
            + "\n\n".join(f"{r.id!r}: {r.assembly}" for r in unmatched.itertuples()),
            type="warning",
        )

    def _norm(x):
        return _REPLICATE_RX.sub("", str(x))

    wrong = out[
        out["_summary_id"].notna()
        & (out["_summary_id"].map(_norm) != out["id"].map(_norm))
    ]
    if not wrong.empty:
        ui.push_message(
            "VAPER: assemblies attached to a different sample than the summary says:\n\n"
            + "\n\n".join(
                f"row {r.id!r} has {r.assembly} (summary id {r._summary_id!r})"
                for r in wrong.itertuples()
            ),
            type="warning",
        )

    return out.drop(columns=["_row", "_stem", "_id_raw", "_summary_id"], errors="ignore")