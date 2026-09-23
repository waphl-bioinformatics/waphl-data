#!/usr/bin/env python3
import os
import re
import sys
import json
import csv
import urllib.parse
import boto3
from botocore.exceptions import ClientError
import pandas as pd
from pathlib import Path
import uuid
from typing import Iterable, Dict, Set, Optional, Any, List, Tuple

from shared import io_ops, data_processing, file_typing
from shared.file_typing import FilePatternMatcher, DEFAULT_TYPE

# ----- Global Setup ----- #
DEST_BUCKET = os.environ.get("DEST_BUCKET")
QUEUE_URL   = os.environ.get("QUEUE_URL")
BATCH_SIZE = 20

SESSION = boto3.session.Session()
S3      = SESSION.client("s3")

SESSION_ID = str(uuid.uuid4().hex)

RAW_PREFIX              = io_ops.RAW_PREFIX
FILES_PREFIX            = io_ops.FILES_PREFIX
FILES_TABLE_KEYS        = data_processing.FILES_TABLE_KEYS
FILES_TABLE_PARTITIONS  = data_processing.FILES_TABLE_PARTITIONS

# ----- Utility ----- #
def log_print(msg: Any) -> None:
    print(str(msg), flush=True)

def parse_uri(s3_uri: str) -> Tuple[str, str]:
    try:
        return file_typing.parse_uri(s3_uri)
    except ValueError as e:
        sys.exit(str(e))

def fastq_columns(fieldnames: Iterable[str]) -> List[str]:
    """Return all manifest column names that start with 'fastq'."""
    return [c for c in (fieldnames or []) if c and c.startswith("fastq")]

def check_file(bucket: str, key: str, fail: bool = True, message: bool = True) -> bool:
    if message:
        log_print(f"Checking exists: s3://{bucket}/{key}")
    try:
        S3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "404":
            if fail:
                sys.exit(f"Missing object: s3://{bucket}/{key}")
            return False
        raise

def list_keys(bucket: str, prefix: str) -> Dict[str, int]:
    """Return {key: last_modified_epoch} for non-directory objects under prefix.

    S3 prefixes are plain string matches, so "runs/VSP025" would also list
    "runs/VSP025B/...". The trailing slash limits the listing to that directory.
    """
    prefix = prefix.rstrip("/") + "/"
    log_print(f"Listing s3://{bucket}/{prefix}")
    paginator = S3.get_paginator("list_objects_v2")
    out: Dict[str, int] = {}
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if not k.endswith("/"):
                out[k] = int(obj["LastModified"].timestamp())
    log_print(f"Keys found: {len(out)}")
    return out

def extract_record(event: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
    log_print("Extracting event record")
    print("Event record:\n", json.dumps(event, indent=2), flush=True)
    detail = event.get("detail", {})
    msgid  = event.get("id")
    bucket = detail.get("bucket", {}).get("name")
    key    = detail.get("object", {}).get("key")
    if key:
        key = urllib.parse.unquote_plus(key)
    if not (msgid and bucket and key):
        sys.exit("Error: Event record malformed (expecting id, bucket.name, object.key)")
    check_file(bucket, key)
    log_print(f"Event OK: id={msgid}, bucket={bucket}, key={key}")
    return bucket, key, msgid

def select_scheme(bucket: str, key: str) -> Dict[str, Any]:
    """Pick a scheme JSON from ./schemes by 'trigger' (filename) and optional 'filter'."""
    trigger = os.path.basename(key)
    log_print(f"Selecting scheme for trigger '{trigger}'")
    scheme_dir = Path("./schemes")
    if not scheme_dir.is_dir():
        sys.exit(f"scheme directory not found: {scheme_dir}")

    for schema_file in scheme_dir.glob("*.json"):
        with open(schema_file, "r") as f:
            data = json.load(f)
        trig = data.get("trigger")
        filt = data.get("filter")
        if trig != trigger:
            continue
        if filt and filt not in key:
            continue
        log_print(f"Using scheme {data.get('scheme', 'null')} from {schema_file.name}")
        return data

    sys.exit("No scheme matched the event trigger")

def get_run_prefix(key: str, scheme: Dict[str, Any]) -> Tuple[str, str]:
    """
    Compute 'base dir' (prefix) and run name from an S3 key.
    Requires 'runs/<run>[/...N levels...]'. Excludes filename.
    """
    dir_level = max(int(scheme.get("dir_level", 0)), 0)
    parts = [p for p in key.split("/") if p]
    try:
        runs_idx = parts.index("runs")
    except ValueError:
        sys.exit(f"Error: 'runs' not found in key: {key}")

    if runs_idx + 1 >= len(parts):
        sys.exit(f"Error: expected 'runs/<run>' structure in key: {key}")

    run = parts[runs_idx + 1]
    is_file = (not key.endswith("/")) and ("." in parts[-1])
    dir_len = len(parts) - (1 if is_file else 0)

    min_end     = runs_idx + 2
    desired_end = runs_idx + 2 + dir_level
    base_end    = max(min_end, min(desired_end, dir_len))
    prefix      = "/".join(parts[:base_end])

    log_print(f"Base dir: {prefix} | run: {run}")
    return prefix, run

def load_run_manifest(bucket: str, base_prefix: str) -> List[Dict[str, str]]:
    man_key = os.path.join(base_prefix, "manifest.csv")
    check_file(bucket, man_key)
    resp = S3.get_object(Bucket=bucket, Key=man_key)
    text = resp["Body"].read().decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    fieldnames = reader.fieldnames or []

    # Require 'sample' plus at least one column starting with 'fastq'.
    missing = {"sample"} - set(fieldnames)
    if missing:
        sys.exit(f"Manifest missing columns: {', '.join(sorted(missing))}")
    if not fastq_columns(fieldnames):
        sys.exit("Manifest missing columns: at least one 'fastq*' column is required")

    rows = list(reader)
    if not rows:
        sys.exit("Manifest is empty")
    return rows

# ----- Matching Helpers ----- #
class SampleMatcher:
    """Find the longest sample name that is fully contained in the input string."""
    def __init__(self, samples: Iterable[str]):
        self.samples = sorted(set(samples), key=len, reverse=True)  # Sort by length descending

    def match(self, text: str) -> Optional[str]:
        """Return the longest sample name contained in text, or None."""
        for sample in self.samples:
            if sample in text:
                return sample
        return None
    
    def match_all(self, texts: List[str]) -> Dict[str, Optional[str]]:
        """Match multiple texts to sample names."""
        return {text: self.match(text) for text in texts}

# ----- Core Logic ----- #
def _dest_key(sample: Optional[str], workflow: str, run: str, key_bn: str, ts: int) -> str:
    return (
        f"{RAW_PREFIX}/id={sample or ''}/"
        f"workflow={workflow}/run={run}/"
        f"file={key_bn}/timestamp={ts}/{key_bn}"
    )

def head_timestamp(bucket: str, key: str) -> int:
    """LastModified epoch for an object; fail if the object is missing."""
    try:
        resp = S3.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "404":
            sys.exit(f"Missing manifest read object: s3://{bucket}/{key}")
        raise
    return int(resp["LastModified"].timestamp())

def classify_keys(
    run_keys: Dict[str, int],
    manifest: List[Dict[str, str]],
    scheme: Dict[str, Any],
    run: str,
    bucket: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """
    Build:
      - meta_rows for the Delta 'files' table
      - transfer_jobs for SQS (source/dest pointers)

    Raw reads are taken directly from the manifest: each fastq* URI keeps its own
    bucket and key, the sample comes from that manifest row, and the file is always
    reportable and typed 'raw_reads'. All other files are discovered by listing the
    run directory (the reads subdirectory is excluded) and are matched to samples /
    typed by pattern as before.
    """
    meta_rows: List[Dict[str, Any]] = []
    transfer_jobs: List[Dict[str, str]] = []

    workflow = scheme.get("workflow")

    # ----- 1) Samples + raw reads from the manifest -----
    samples = sorted({row["sample"] for row in manifest if row.get("sample")})
    sm = SampleMatcher(samples)

    # (sample, bucket, key) for every fastq* cell in the manifest
    manifest_reads: List[Tuple[Optional[str], str, str]] = []
    manifest_read_keys: Set[str] = set()
    for row in manifest:
        sample = row.get("sample")
        for col in fastq_columns(row.keys()):
            if row.get(col):
                b, k = parse_uri(row[col])
                manifest_reads.append((sample, b, k))
                manifest_read_keys.add(k)

    # ----- 2) Emit raw-read rows/jobs straight from the manifest -----
    for sample, src_bucket, key in manifest_reads:
        key_bn = os.path.basename(key)
        ts = head_timestamp(src_bucket, key)
        dest_key = _dest_key(sample, workflow, run, key_bn, ts)

        meta_rows.append({
            "id": sample,                       # manifest's sample designation
            "workflow": workflow,
            "run": run,
            "file": key_bn,
            "timestamp": ts,
            "reportable": True,
            "type": "raw_reads",
            "origin": f"s3://{src_bucket}/{key}",   # manifest's own bucket
            "current": f"s3://{DEST_BUCKET}/{dest_key}",
        })

        if not check_file(DEST_BUCKET, dest_key, fail=False, message=False):
            transfer_jobs.append({
                "SOURCE_BUCKET": src_bucket,
                "SOURCE_KEY": key,
                "DEST_BUCKET": DEST_BUCKET,
                "DEST_KEY": dest_key,
            })

    # ----- 3) Everything else comes from the run directory listing -----
    reportable_files = scheme.get("reportable_files", {})
    fp = FilePatternMatcher(reportable_files)
    if fp.skipped:
        log_print(f"Ignoring {len(fp.skipped)} reportable_files entries with no usable pattern")

    for key, ts in run_keys.items():
        if f"{run}/reads/" in key:
            continue

        key_bn = os.path.basename(key)
        sample = sm.match(key)

        ftype = fp.match_type(key_bn)
        reportable = ftype is not None
        ftype = ftype or DEFAULT_TYPE

        dest_key = _dest_key(sample, workflow, run, key_bn, ts)

        meta_rows.append({
            "id": sample,
            "workflow": workflow,
            "run": run,
            "file": key_bn,
            "timestamp": ts,
            "reportable": bool(reportable),
            "type": ftype,
            "origin": f"s3://{bucket}/{key}",       # function's bucket
            "current": f"s3://{DEST_BUCKET}/{dest_key}",
        })

        if not check_file(DEST_BUCKET, dest_key, fail=False, message=False):
            transfer_jobs.append({
                "SOURCE_BUCKET": bucket,
                "SOURCE_KEY": key,
                "DEST_BUCKET": DEST_BUCKET,
                "DEST_KEY": dest_key,
            })

    return meta_rows, transfer_jobs

def push_meta(meta_rows: List[Dict[str, Any]]) -> None:
    """Write metadata rows to the Delta 'files' table."""
    if not meta_rows:
        log_print("No metadata rows to write.")
        return

    meta_rows_standard = []
    for row in meta_rows:
        row_standard = data_processing.standardize_data(data=row, default_columns=FILES_TABLE_KEYS)
        meta_rows_standard.append(row_standard)

    # Build DF
    df = pd.DataFrame(meta_rows_standard)
    required_first = FILES_TABLE_KEYS + [c for c in df.columns if c not in FILES_TABLE_KEYS]
    df = df[required_first]

    df = data_processing.clean_dataframe(df)

    io_ops.write_delta(
        df=df,
        uri=f"s3://{DEST_BUCKET}/{FILES_PREFIX}",
        key_cols=FILES_TABLE_KEYS,
        partition_by=FILES_TABLE_PARTITIONS,
    )
    log_print(f"Wrote {len(df)} rows to Delta: s3://{DEST_BUCKET}/{FILES_PREFIX}")


def chunked(iterable: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]

def send_to_sqs(jobs: List[Dict[str, str]], queue_url: str, batch_size: int = 10) -> None:
    """Send transfer jobs in batches to SQS (one JSON list per message)."""
    if not jobs:
        log_print("No transfer jobs to enqueue.")
        return
    sqs = boto3.client("sqs")
    log_print(f"Enqueuing {len(jobs)} jobs to SQS in batches of {batch_size}")
    for batch in chunked(jobs, batch_size):
        body = json.dumps(batch)
        resp = sqs.send_message(QueueUrl=queue_url, MessageBody=body)
        log_print(f"Sent batch: MessageId={resp.get('MessageId')} (size={len(batch)})")

# ----- Lambda Handler ----- #
def handler(event, context):
    bucket, key, _msgid = extract_record(event)
    scheme = select_scheme(bucket, key)
    run_prefix, run_name = get_run_prefix(key, scheme)
    manifest = load_run_manifest(bucket, run_prefix)
    run_keys = list_keys(bucket, run_prefix)

    meta_rows, transfer_jobs = classify_keys(run_keys, manifest, scheme, run_name, bucket)

    try:
        push_meta(meta_rows)
    except Exception as e:
        log_print(f"Delta write failed: {e}")
        sys.exit(1)
    
    if transfer_jobs:
        try:
            send_to_sqs(transfer_jobs, QUEUE_URL, batch_size=BATCH_SIZE)
        except Exception as e:
            log_print(f"SQS enqueue failed: {e}")
            sys.exit(1)
    else:
        log_print("All files already exist at the destination. Nothing to transfer.")

# ----- Optional Local Test ----- #
if __name__ == "__main__":
    with open("/data/test_event.json") as f:
        test_event = json.load(f)
    handler(test_event, None)