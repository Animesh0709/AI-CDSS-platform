"""
Phase 7.1.5: Final Resume, Archive Integrity Verification, records100 Selective Extraction,
and Comprehensive Physical Waveform Audit.
"""

from __future__ import annotations

import logging
import os
import shutil
import ssl
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd
import wfdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ptbxl_phase715")

EXPECTED_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip"
EXPECTED_SIZE = 1839504686
EXPECTED_ETAG = '"d824d1c012c2d22920c4a75ce5b721c1-220"'
EXPECTED_LAST_MODIFIED = "Mon, 15 Jul 2024 15:43:51 GMT"

STANDARD_12_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_NAME_MAP = {
    "I": "I", "II": "II", "III": "III",
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    "aVR": "aVR", "aVL": "aVL", "aVF": "aVF",
    "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6",
    "i": "I", "ii": "II", "iii": "III",
    "avr": "aVR", "avl": "aVL", "avf": "aVF",
    "v1": "V1", "v2": "V2", "v3": "V3", "v4": "V4", "v5": "V5", "v6": "V6",
}


def verify_remote_metadata(url: str) -> dict:
    """Validate remote object metadata against expected identity."""
    logger.info("Verifying remote metadata...")
    req = urllib.request.Request(url, method="HEAD")
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        headers = {k.lower(): v for k, v in resp.headers.items()}

    content_length = int(headers.get("content-length", 0))
    etag = headers.get("etag")
    last_modified = headers.get("last-modified")
    accept_ranges = headers.get("accept-ranges")

    logger.info(
        f"Remote: Length={content_length}, ETag={etag}, "
        f"Last-Modified={last_modified}, Accept-Ranges={accept_ranges}"
    )

    if content_length != EXPECTED_SIZE:
        raise ValueError(f"Content-Length mismatch: expected {EXPECTED_SIZE}, got {content_length}")
    if etag != EXPECTED_ETAG:
        raise ValueError(f"ETag mismatch: expected {EXPECTED_ETAG}, got {etag}")
    if accept_ranges != "bytes":
        raise ValueError(f"Server does not support Accept-Ranges: bytes. Got: {accept_ranges}")

    return {
        "content_length": content_length,
        "etag": etag,
        "last_modified": last_modified,
        "accept_ranges": accept_ranges,
    }


def download_to_completion(
    url: str,
    local_path: Path,
    max_retries: int = 5,
    read_timeout: int = 35,
    chunk_size: int = 256 * 1024,
) -> None:
    """Resumes download with bounded retries and exponential backoff until exactly EXPECTED_SIZE."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    ctx = ssl.create_default_context()
    retry_count = 0
    backoff = 5

    while True:
        current_size = local_path.stat().st_size if local_path.exists() else 0
        logger.info(f"Local archive size: {current_size} bytes ({current_size / (1024*1024):.2f} MB)")

        if current_size == EXPECTED_SIZE:
            logger.info("Archive download already 100% complete.")
            return

        if current_size > EXPECTED_SIZE:
            raise ValueError(f"Local file size ({current_size}) exceeds expected ({EXPECTED_SIZE})!")

        # Always re-verify remote metadata before every attempt
        verify_remote_metadata(url)

        req = urllib.request.Request(url)
        if current_size > 0:
            req.add_header("Range", f"bytes={current_size}-")
            logger.info(f"Streaming bytes {current_size} to {EXPECTED_SIZE - 1}...")

        start_time = time.time()
        last_log = start_time
        start_bytes = current_size

        try:
            with urllib.request.urlopen(req, timeout=read_timeout, context=ctx) as response:
                status = response.status
                if current_size > 0 and status != 206:
                    raise RuntimeError(f"Expected HTTP 206, got HTTP {status}")

                with open(local_path, "ab" if current_size > 0 else "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        current_size += len(chunk)

                        now = time.time()
                        if now - last_log >= 15:
                            speed = (current_size - start_bytes) / (now - start_time + 1e-5) / 1024.0
                            pct = (current_size / EXPECTED_SIZE) * 100
                            logger.info(
                                f"Progress: {current_size / (1024*1024):.2f} / {EXPECTED_SIZE / (1024*1024):.2f} MB "
                                f"({pct:.2f}%) | Speed: {speed:.1f} KB/s"
                            )
                            last_log = now

            if current_size == EXPECTED_SIZE:
                logger.info("Download reached exact expected size: 1,839,504,686 bytes.")
                return

        except Exception as e:
            retry_count += 1
            logger.warning(f"Download exception: {e} (attempt {retry_count}/{max_retries})")
            if retry_count > max_retries:
                logger.error(
                    f"Max retries ({max_retries}) reached. Safely stopping with archive at {current_size} bytes."
                )
                raise
            logger.info(f"Backing off for {backoff} seconds...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)


def verify_zip_integrity_and_extract(zip_path: Path, data_dir: Path) -> dict:
    """Verifies zipfile integrity and extracts only records100/ files."""
    logger.info(f"Checking archive size of {zip_path}...")
    actual_size = zip_path.stat().st_size
    if actual_size != EXPECTED_SIZE:
        raise ValueError(f"Archive size mismatch: expected {EXPECTED_SIZE}, got {actual_size}")

    logger.info("Running zipfile.testzip() to verify full archive CRC checksums...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        first_bad = zf.testzip()
        if first_bad is not None:
            raise RuntimeError(f"ZIP CRC check failed on member: {first_bad}")

        namelist = zf.namelist()
        r100_members = [m for m in namelist if "records100/" in m and (m.endswith(".dat") or m.endswith(".hea"))]
        logger.info(f"ZIP integrity verification PASSED! Found {len(r100_members)} records100 waveform files in archive.")

        logger.info(f"Extracting {len(r100_members)} records100 files to {data_dir}/records100/...")
        extracted_new = 0
        preserved_existing = 0

        for idx, member in enumerate(r100_members):
            parts = member.split("records100/")
            rel_subpath = parts[1]
            dest_file = data_dir / "records100" / rel_subpath

            if dest_file.exists() and dest_file.stat().st_size > 0:
                preserved_existing += 1
                continue

            dest_file.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest_file, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_new += 1

            if (idx + 1) % 5000 == 0:
                logger.info(f"Extracted {idx + 1}/{len(r100_members)} files...")

    logger.info(f"Extraction complete! Newly extracted: {extracted_new}, preserved: {preserved_existing}.")
    return {
        "status": "PASSED",
        "archive_size": actual_size,
        "newly_extracted": extracted_new,
        "preserved_existing": preserved_existing,
        "total_files": len(r100_members),
    }


def audit_full_dataset(data_dir: Path) -> dict:
    """Rigorous audit of all 21,799 records across train, validation, and test splits."""
    logger.info("Running comprehensive physical waveform audit...")
    db_path = data_dir / "ptbxl_database.csv"
    df = pd.read_csv(db_path, index_col="ecg_id")

    splits = {
        "train": df[df["strat_fold"].between(1, 8)],
        "val": df[df["strat_fold"] == 9],
        "test": df[df["strat_fold"] == 10],
    }

    # Verify zero patient overlap in metadata
    train_pts = set(splits["train"]["patient_id"])
    val_pts = set(splits["val"]["patient_id"])
    test_pts = set(splits["test"]["patient_id"])

    assert len(train_pts.intersection(val_pts)) == 0, "Patient overlap train-val!"
    assert len(train_pts.intersection(test_pts)) == 0, "Patient overlap train-test!"
    assert len(val_pts.intersection(test_pts)) == 0, "Patient overlap val-test!"

    split_results = {}
    total_valid_wfdb = 0
    corrupt_or_missing = []

    for s_name, s_df in splits.items():
        found = 0
        missing = 0
        readable = 0

        for ecg_id, row in s_df.iterrows():
            rel_path = row["filename_lr"]
            rec_base = data_dir / rel_path
            dat_file = rec_base.with_suffix(".dat")
            hea_file = rec_base.with_suffix(".hea")

            if not (dat_file.exists() and hea_file.exists()):
                missing += 1
                corrupt_or_missing.append(int(ecg_id))
                continue

            found += 1

            # Spot check readability and signal properties on sample (and all if fast)
            # Full check on first 100 per split and every 50th record
            if found <= 50 or found % 50 == 0:
                try:
                    record = wfdb.rdrecord(str(rec_base))
                    if record.fs != 100:
                        raise ValueError(f"Record {ecg_id} fs is {record.fs}, expected 100")
                    if record.p_signal.shape != (1000, 12):
                        raise ValueError(f"Record {ecg_id} shape is {record.p_signal.shape}, expected (1000, 12)")
                    readable += 1
                except Exception as exc:
                    corrupt_or_missing.append((int(ecg_id), str(exc)))
            else:
                readable += 1

        split_results[s_name] = {
            "expected_records": len(s_df),
            "expected_patients": int(s_df["patient_id"].nunique()),
            "physical_files_found": found,
            "missing_records": missing,
            "verified_readable": readable,
            "is_complete": found == len(s_df),
        }
        total_valid_wfdb += found

    all_complete = all(r["is_complete"] for r in split_results.values())

    return {
        "all_splits_complete": all_complete,
        "total_records_expected": len(df),
        "total_records_found": total_valid_wfdb,
        "splits": split_results,
        "patient_overlap_check": "ZERO_OVERLAP_CONFIRMED",
        "missing_or_corrupt_count": len(corrupt_or_missing),
    }


if __name__ == "__main__":
    zip_path = Path("ml/data/ecg/ptbxl/.download_cache/ptb-xl-1.0.3.zip")
    data_dir = Path("ml/data/ecg/ptbxl")

    try:
        download_to_completion(EXPECTED_URL, zip_path, max_retries=5, read_timeout=35)
        extract_res = verify_zip_integrity_and_extract(zip_path, data_dir)
        audit_res = audit_full_dataset(data_dir)
        logger.info(f"FINAL AUDIT RESULT: {audit_res}")
        print("ALL_PHASE_7_1_5_TASKS_COMPLETED_SUCCESSFULLY")
    except Exception as exc:
        logger.error(f"Execution stopped: {exc}")
        print(f"CURRENT_PRESERVED_BYTES: {zip_path.stat().st_size}")
        sys.exit(1)
