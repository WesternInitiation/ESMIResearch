#!/usr/bin/env python3
"""
Scan a demo TAR in GCS and upload manifest.json so Vercel can list members
without downloading the whole archive.

Usage (Cloud Shell / machine with gcloud ADC):

  python3 cloud_run/write_demo_manifest.py
  python3 cloud_run/write_demo_manifest.py \\
      --bucket esmi-research-demo-data \\
      --archive LC09_L2SP_016030_20260526_20260527_02_T1.tar

Requires: pip install google-cloud-storage
  and ADC (`gcloud auth application-default login` or Cloud Shell).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Repo root on sys.path so `image_io` imports when run from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from image_io import is_tar_archive, scan_archive_image_entries  # noqa: E402


DEFAULT_BUCKET = "esmi-research-demo-data"
DEFAULT_ARCHIVE = "LC09_L2SP_016030_20260526_20260527_02_T1.tar"


def build_manifest(
    archive: str, entries: list[dict], *, bucket: str | None = None
) -> dict:
    members = sorted(e["name"] for e in entries)
    payload: dict = {
        "archive": archive,
        "members": members,
    }
    if bucket:
        payload["bucket"] = bucket
    # Include byte ranges when available (uncompressed .tar) for ranged GETs.
    ranged = [
        {"name": e["name"], "offset": e["offset"], "size": e["size"]}
        for e in entries
        if "offset" in e
    ]
    if ranged:
        payload["entries"] = sorted(ranged, key=lambda e: e["name"])
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--manifest-name",
        default="manifest.json",
        help="Object name for the uploaded manifest (default: manifest.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print manifest JSON without uploading",
    )
    args = parser.parse_args()

    if not is_tar_archive(args.archive):
        print(f"Not a TAR archive: {args.archive}", file=sys.stderr)
        return 2

    from google.cloud import storage

    client = storage.Client()
    blob = client.bucket(args.bucket).blob(args.archive)
    if not blob.exists():
        print(f"Missing gs://{args.bucket}/{args.archive}", file=sys.stderr)
        return 1

    size_mb = (blob.size or 0) / (1024 * 1024)
    print(
        f"Streaming gs://{args.bucket}/{args.archive} ({size_mb:.0f} MB) for member list…",
        flush=True,
    )

    with blob.open("rb") as handle:
        entries = scan_archive_image_entries(handle, filename=args.archive)

    manifest = build_manifest(args.archive, entries)
    text = json.dumps(manifest, indent=2) + "\n"
    print(f"Found {len(manifest['members'])} image members.", flush=True)
    if args.dry_run:
        print(text)
        return 0

    out = client.bucket(args.bucket).blob(args.manifest_name)
    out.upload_from_string(text, content_type="application/json")
    print(f"Uploaded gs://{args.bucket}/{args.manifest_name}")
    if "entries" in manifest:
        print("Includes byte offsets — Vercel can Range-GET a single band.")
    else:
        print(
            "No byte offsets (compressed TAR). Listing works; extracts still need "
            "a full stream via Cloud Run /v1/demo/extract.",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
