"""Resumable image downloader for the Airbnb listings CSV

Features
- Resumes: skips files already downloaded (checks metadata and on-disk file)
- Safe writes: stream to `.part` then rename on success
- Concurrent downloads with ThreadPoolExecutor
- Optional post-download resize (Pillow) to a target size (e.g. 224x224)
- Writes `images/metadata.csv` with one row per URL downloaded/tried
- Uses `id` column if present to group images by listing; falls back to row index

Usage examples
- Quick run (resume-aware):
    python scripts/download_images.py

- Limit to 8 workers and resize to 224×224:
    python scripts/download_images.py --workers 8 --resize 224

Notes
- The script downloads whatever the server serves. Some CDNs support size/query params but this script does not modify URLs; instead it can downscale after download.
- By default it filters listings the same way as the EDA (50 < price < 1000). Use --no-price-filter to keep all rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from PIL import Image

# Defaults
ROOT = Path(__file__).resolve().parents[1]
CSV_PATH_DEFAULT = ROOT / "listings.csv"
OUT_DIR_DEFAULT = ROOT / "images"
METADATA_NAME = "metadata.csv"

# Small helper
_lock = threading.Lock()


def clean_price(s):
    if s is None:
        return None
    try:
        if isinstance(s, (int, float)):
            return float(s)
        return float(str(s).replace("$", "").replace(",", "").strip())
    except Exception:
        return None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def url_basename(url: str) -> str:
    # pick a stable basename from URL path; fall back to sha1 if weird
    try:
        p = Path(url.split("?")[0])
        name = p.name
        if name == "":
            # fallback
            h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
            return f"image_{h}.jpg"
        return name
    except Exception:
        return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12] + ".jpg"


def load_metadata(outdir: Path) -> Dict[str, Dict]:
    meta_file = outdir / METADATA_NAME
    if not meta_file.exists():
        return {}
    d = {}
    with meta_file.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            d[r["url"]] = r
    return d


def append_metadata_row(outdir: Path, row: Dict):
    meta_file = outdir / METADATA_NAME
    write_header = not meta_file.exists()
    with _lock:
        with meta_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def safe_download(url: str, dest: Path, timeout: int = 20) -> Tuple[bool, Optional[int], Optional[str]]:
    """Download URL to dest.part then rename to dest. Returns (ok, bytes, http_status)"""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            status = r.status_code
            if r.status_code != 200:
                return False, None, str(status)
            total = 0
            ensure_dir(dest.parent)
            with tmp.open("wb") as fd:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        fd.write(chunk)
                        total += len(chunk)
            tmp.replace(dest)
            return True, total, str(status)
    except Exception as e:
        # cleanup partial file on failure
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False, None, str(e)


def resize_image(path: Path, size: int):
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = im.resize((size, size), Image.BILINEAR)
            im.save(path, quality=90)
        return True
    except Exception:
        return False


def prepare_download_tasks(csv_path: Path, outdir: Path, price_filter: bool = True):
    import pandas as pd

    df = pd.read_csv(csv_path)

    # replicate EDA filter by default
    if price_filter and "price" in df.columns:
        df["price_clean"] = df["price"].apply(clean_price)
        df = df[(df["price_clean"] > 50) & (df["price_clean"] < 1000)].copy()

    tasks = []
    for idx, row in df.iterrows():
        url = None
        # prefer a column named picture_url, else look for image columns pattern
        if "picture_url" in row and not pd.isna(row["picture_url"]):
            url = row["picture_url"]
            img_index = 0
            listing_id = str(row.get("id", idx))
            basename = url_basename(url)
            dest = outdir / listing_id / f"{img_index:03d}_{basename}"
            tasks.append({"listing_id": listing_id, "url": url, "dest": dest})
        else:
            # try to find columns that look like image URLs (common patterns)
            for c in row.index:
                if "picture" in str(c).lower() or "image" in str(c).lower():
                    v = row[c]
                    if isinstance(v, str) and v.strip().startswith("http"):
                        url = v.strip()
                        listing_id = str(row.get("id", idx))
                        basename = url_basename(url)
                        dest = outdir / listing_id / f"0_{basename}"
                        tasks.append({"listing_id": listing_id, "url": url, "dest": dest})
                        break
    return tasks


def download_worker(task, outdir: Path, meta: Dict, resize: Optional[int], timeout: int, overwrite: bool, delay: float) -> Dict:
    url = task["url"]
    dest: Path = task["dest"]
    dest_exists = dest.exists()

    # If metadata says done, skip
    if url in meta and meta[url].get("status") == "done" and dest_exists and not overwrite:
        return {"url": url, "status": "skipped", "file": str(dest), "note": "already done"}

    # If file exists and not overwrite, assume OK
    if dest_exists and not overwrite:
        size = dest.stat().st_size
        append_metadata_row(outdir, {"timestamp": datetime.utcnow().isoformat(), "url": url, "file": str(dest), "status": "done", "http_status": "exists", "bytes": size})
        return {"url": url, "status": "skipped", "file": str(dest), "note": "file exists"}

    ok, nbytes, http_status = safe_download(url, dest, timeout=timeout)
    row = {"timestamp": datetime.utcnow().isoformat(), "url": url, "file": str(dest), "status": "done" if ok else "failed", "http_status": http_status or "", "bytes": nbytes or 0}
    append_metadata_row(outdir, row)

    if ok and resize:
        succeeded = resize_image(dest, resize)
        if not succeeded:
            return {"url": url, "status": "done", "file": str(dest), "note": "resize_failed"}
    # optional polite delay
    if delay:
        time.sleep(delay)
    return {"url": url, "status": "done" if ok else "failed", "file": str(dest)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(CSV_PATH_DEFAULT))
    p.add_argument("--outdir", default=str(OUT_DIR_DEFAULT))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--resize", type=int, default=0, help="Resize downloaded images to SxS (e.g. 224); 0=no resize")
    p.add_argument("--timeout", type=int, default=30)
    p.add_argument("--no-price-filter", dest="price_filter", action="store_false", help="Do not apply the 50<price<1000 filter used by EDA")
    p.add_argument("--overwrite", action="store_true", help="Redownload and overwrite existing files")
    p.add_argument("--delay", type=float, default=0.0, help="Delay (seconds) between downloads to be polite")
    p.add_argument("--duration-hours", type=float, default=0.0, help="Spread downloads evenly over N hours (0 = no pacing). When set, the script forces sequential downloads and computes an interval.")
    p.add_argument("--min-interval", type=float, default=10.0, help="Minimum seconds between downloads when pacing (applies when --duration-hours > 0)")
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    print(f"Preparing tasks from CSV: {args.csv}")
    tasks = prepare_download_tasks(Path(args.csv), outdir, price_filter=args.price_filter)
    print(f"Found {len(tasks)} image tasks (duplicates may exist).")

    meta = load_metadata(outdir)

    # dedupe by URL, keep first task for a given URL
    unique_tasks: Dict[str, Dict] = {}
    for t in tasks:
        if t["url"] not in unique_tasks:
            unique_tasks[t["url"]] = t

    tasks = list(unique_tasks.values())
    n_tasks = len(tasks)
    print(f"After deduplication: {n_tasks} unique URLs to process.")

    # If user requested pacing over a duration, compute per-task interval and force sequential mode
    if args.duration_hours and n_tasks > 0:
        total_seconds = args.duration_hours * 3600.0
        computed_interval = total_seconds / float(n_tasks)
        interval = max(args.min_interval, computed_interval)
        print(f"Pacing enabled: spreading {n_tasks} downloads over {args.duration_hours}h -> interval {interval:.1f}s (min {args.min_interval}s). Forcing sequential downloads.")
        args.delay = interval
        args.workers = 1

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [ex.submit(download_worker, t, outdir, meta, args.resize or None, args.timeout, args.overwrite, args.delay) for t in tasks]
        for f in as_completed(futures):
            try:
                r = f.result()
            except Exception as e:
                r = {"status": "error", "error": str(e)}
            results.append(r)
            # lightweight stdout progress
            done = sum(1 for x in results if x.get("status") in ("done", "skipped"))
            total = len(tasks)
            sys.stdout.write(f"\rProgress: {done}/{total} ({done/total:.0%})")
            sys.stdout.flush()
    print("\nAll download workers finished.")


if __name__ == "__main__":
    main()
