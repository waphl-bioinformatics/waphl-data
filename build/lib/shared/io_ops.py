from __future__ import annotations

import json
import os
import random
import time
from typing import Iterable, Optional, Sequence, Union

import boto3
import numpy as np
import pandas as pd
import pyarrow as pa
import streamlit as st
from botocore.exceptions import ClientError
from deltalake import DeltaTable, write_deltalake

from shared.aws_ops import log_print

try:
    from deltalake.exceptions import CommitFailedError
except Exception:
    CommitFailedError = Exception

# -----------------
# Constants / Paths
# -----------------

RAW_PREFIX         = "data"
FILES_PREFIX       = "tables/metadata/"
RESULTS_PREFIX     = "tables/results/"
TERRA_CACHE_PREFIX = "tables/cache/terra/"


# --------------
# Arrow Helpers
# --------------

def _to_arrow(
    records_or_df: Union[pd.DataFrame, Sequence[dict]],
    expected_cols: Optional[Iterable[str]] = None,
) -> pa.Table:
    """
    Convert records or a DataFrame to a PyArrow table with light type hygiene.
    Ensures:
      - boolean dtype for ('reportable', 'inspected') if present
      - int64 dtype for ('timestamp', 'last_run_time') if present
      - adds any `expected_cols` missing (with sensible defaults)
      - stable column order if `expected_cols` provided
    """
    df = records_or_df if isinstance(records_or_df, pd.DataFrame) else pd.DataFrame(records_or_df)

    # Type hygiene
    for c in ("reportable", "inspected"):
        if c in df.columns:
            df[c] = df[c].astype(bool)
    for c in ("timestamp", "last_run_time"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

    # Enforce stable column order & presence
    if expected_cols:
        for col in expected_cols:
            if col not in df.columns:
                if col in ("reportable", "inspected", "file_exists"):
                    df[col] = False
                elif col in ("timestamp", "last_run_time"):
                    df[col] = 0
                else:
                    df[col] = ""
        df = df[list(expected_cols)]

    return pa.Table.from_pandas(df, preserve_index=False)


# ---------------------------------------------
# JSON normalization for complex dtype columns
# ---------------------------------------------

def _jsonify_if_complex(x):
    """
    Convert complex objects into compact, deterministic JSON (or scalars).
    Leaves plain scalars/strings as-is. Numpy scalars become Python scalars.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None

    # Treat bytes as UTF-8 strings if present
    if isinstance(x, (bytes, bytearray)):
        try:
            return x.decode("utf-8")
        except Exception:
            return str(x)

    # Normalize numpy scalars
    if isinstance(x, np.generic):
        return x.item()

    # numpy arrays -> lists -> JSON
    if isinstance(x, np.ndarray):
        return json.dumps(x.tolist(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    # Lists / tuples / sets / dicts -> JSON (sets become lists)
    if isinstance(x, (list, tuple, set, dict)):
        if isinstance(x, set):
            x = list(x)
        return json.dumps(x, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    return x


def _normalize_lists_to_json_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each column, if any value is list/tuple/dict/set/ndarray, convert ALL such
    values in that column to compact JSON strings. Keep None/NaN as None.
    """
    df = df.copy()
    for col in df.columns:
        s = df[col]
        needs_json = False

        # Quick sample check to avoid unnecessary mapping over entire column
        sample = s.values[:100] if len(s) > 100 else s.values
        for v in sample:
            if isinstance(v, (list, tuple, set, dict, np.ndarray)):
                needs_json = True
                break

        if needs_json:
            df[col] = s.map(_jsonify_if_complex)
        else:
            # Light pass to clean numpy scalars / bytes in object cols
            if s.dtype == "object":
                df[col] = s.map(_jsonify_if_complex)
    return df


# -----------------------
# Delta Lake I/O helpers
# -----------------------

def _is_conflict_error(e: Exception) -> bool:
    return "concurrent transactions added new data" in str(e).lower()

def write_delta(
    df: pd.DataFrame,
    uri: str,
    key_cols: Optional[Iterable[str]] = None,
    partition_by: Optional[Iterable[str]] = None,
    schema_mode: str = "merge",
    storage_options: Optional[dict] = None,
    max_retries: int = 5,
    retry_base_sleep: float = 1.0,
) -> None:
    """
    Upsert/append a DataFrame to a Delta table at `uri`.

    - Normalizes complex columns to JSON strings
    - Creates table on first write (append)
    - If `key_cols` are provided and present, perform MERGE (update/insert)
      else append
    - Retries MERGE on optimistic concurrency conflicts

    Args:
        df: DataFrame to write
        uri: Delta table URI
        key_cols: Columns to use as merge keys
        partition_by: Columns to partition by
        schema_mode: Schema evolution mode - "merge", "overwrite", or None
        storage_options: Storage backend options (S3/GCS/Azure/etc.)
        max_retries: Max retries for concurrent commit failures
        retry_base_sleep: Base sleep in seconds for exponential backoff
    """
    storage_options = storage_options or {}
    key_cols = list(key_cols) if key_cols else []

    df = _normalize_lists_to_json_strings(df)
    at = _to_arrow(df)

    # Check if table exists
    try:
        DeltaTable(uri, storage_options=storage_options)
        table_exists = True
    except Exception:
        table_exists = False

    # Append if new table or no merge keys
    should_append = not table_exists or not key_cols or not set(key_cols).issubset(df.columns)
    if should_append:
        write_deltalake(
            uri,
            at,
            mode="append",
            partition_by=partition_by,
            schema_mode=schema_mode,
            storage_options=storage_options,
        )
        return

    # MERGE with retry on concurrent commit conflicts
    predicate = " AND ".join([f"target.{c} = source.{c}" for c in key_cols])

    for attempt in range(1, max_retries + 1):
        try:
            (
                DeltaTable(uri, storage_options=storage_options)
                .merge(source=at, predicate=predicate, source_alias="source", target_alias="target")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute()
            )
            return

        except Exception as e:
            if not _is_conflict_error(e) or attempt == max_retries:
                raise

            sleep_s = retry_base_sleep * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            log_print(f"Delta commit conflict at {uri}, retrying in {sleep_s:.1f}s (attempt {attempt}/{max_retries})")
            time.sleep(sleep_s)

_FILTER_OPS = {
    "=":      lambda s, v: s == v,
    "==":     lambda s, v: s == v,
    "!=":     lambda s, v: s != v,
    "<":      lambda s, v: s < v,
    "<=":     lambda s, v: s <= v,
    ">":      lambda s, v: s > v,
    ">=":     lambda s, v: s >= v,
    "in":     lambda s, v: s.isin(list(v)),
    "not in": lambda s, v: ~s.isin(list(v)),
}


def read_delta_as_pandas(uri, filters=None, columns=None, storage_options=None):
    """
    Read a Delta table into pandas with DNF-style filters [(col, op, value), ...].

    Only partition-column filters are pushed down to deltalake. Filters on
    regular columns are applied in pandas after the read: files written by a
    deltalake MERGE (DataFusion) store strings as string_view, and pyarrow's
    dataset filter can't compare those against a plain string literal
    ("Function 'equal' has no kernel matching input types (string, string_view)").
    """
    dt = DeltaTable(uri, storage_options=storage_options)
    filters = [tuple(f) for f in (filters or [])]

    partition_cols = set(dt.metadata().partition_columns)
    part_filters = [f for f in filters if f[0] in partition_cols]
    row_filters = [f for f in filters if f[0] not in partition_cols]

    for col, op, _ in row_filters:
        if op not in _FILTER_OPS:
            raise ValueError(f"Unsupported filter operator {op!r} on column {col!r}")

    read_cols = None
    if columns:
        read_cols = list(dict.fromkeys(list(columns) + [c for c, _, _ in row_filters]))

    df = dt.to_pandas(filters=part_filters or None, columns=read_cols)

    if row_filters:
        mask = pd.Series(True, index=df.index)
        for col, op, val in row_filters:
            if col not in df.columns:
                raise ValueError(f"Filter column {col!r} not found in table")
            mask &= _FILTER_OPS[op](df[col], val).fillna(False).astype(bool)
        df = df[mask].reset_index(drop=True)

    if columns:
        df = df[list(columns)]
    return df

def delta_partition_values(uri, partition_col, storage_options=None):
    dt = DeltaTable(uri, storage_options=storage_options)
    files = dt.get_add_actions(flatten=True)
    col = f"partition.{partition_col}"
    return sorted(set(files[col].to_pylist()))


# ----------------
# S3 List Helpers
# ----------------

def list_prefix_dirs(bucket: str, prefix: str) -> list[str]:
    """
    Return sorted list of immediate directory names under an S3 prefix.
    """
    s3 = boto3.client("s3")
    prefix = str(prefix).rstrip("/") + "/"

    paginator = s3.get_paginator("list_objects_v2")
    dirs = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            full_prefix = cp["Prefix"]
            name = os.path.basename(full_prefix.rstrip("/"))
            if name:
                dirs.add(name)

    return sorted(dirs)


def list_s3_keys(bucket: str, prefix: str):
    """
    Yield all object keys under a given S3 prefix (handles pagination).
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def s3_file_exists(bucket: str, key: str) -> bool:
    """
    Check if a file exists in an S3 bucket.

    Args:
        bucket: S3 bucket name.
        key: Full key (path) of the file within the bucket.

    Returns:
        True if file exists, False otherwise.
    """
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        # 404 means the object does not exist
        if e.response.get("Error", {}).get("Code") == "404":
            return False
        # Re-raise other errors (e.g., permission issues)
        raise