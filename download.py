"""Download public Allen Visual Coding Neuropixels data without AllenSDK."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import time
import requests

API = "https://api.brain-map.org"
DEFAULT_SESSION = 791319847


def query(criteria):
    response = requests.get(f"{API}/api/v2/data/query.json",
                            params={"criteria": criteria, "num_rows": "all"}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"Allen API query failed: {payload.get('msg')}")
    return payload["msg"]


def download(url, destination):
    """Resume a partial download; install the final file only when complete."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(4):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("Server returned an unexpected byte range")
                length = response.headers.get("Content-Length")
                expected = int(length) if length else None
                written = 0
                report_at = 256 * 1024 * 1024
                with partial.open("ab" if append else "wb") as output:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if not chunk:
                            continue
                        output.write(chunk)
                        written += len(chunk)
                        if written >= report_at:
                            print(f"{destination.name}: {(written + (offset if append else 0))/1e9:.2f} GB", flush=True)
                            report_at += 256 * 1024 * 1024
                if expected is not None and written != expected:
                    raise IOError(f"Incomplete download: {written} of {expected} bytes")
            if partial.stat().st_size == 0:
                raise IOError("Downloaded an empty file")
            partial.replace(destination)
            return destination
        except (requests.RequestException, IOError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def fetch(session=DEFAULT_SESSION, root="data", frames=(0, 30, 60, 90)):
    root = Path(root)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rows = query("model::WellKnownFile,rma::criteria,"
                 "well_known_file_type[name$eq'EcephysNwb']"
                 f"[attachable_type$eq'EcephysSession'][attachable_id$eq{int(session)}]")
    if len(rows) != 1:
        raise ValueError(f"Expected one NWB file for session {session}; got {len(rows)}")
    url = API + rows[0]["download_link"]
    nwb = download(url, raw / f"session_{session}.nwb")
    with nwb.open("rb") as handle:
        if handle.read(8) != b"\x89HDF\r\n\x1a\n":
            raise ValueError(f"Not an HDF5/NWB file: {nwb}")
    templates = query("model::WellKnownFile,rma::criteria,"
                      "well_known_file_type[name$eq'Stimulus']"
                      "[attachable_type$eq'Product'][attachable_id$eq714914585]")
    by_frame = {}
    for row in templates:
        match = re.search(r"/(\d+)\.tiff$", row["path"])
        if match:
            by_frame[int(match.group(1))] = row
    images = {}
    for frame in frames:
        if frame not in by_frame:
            raise ValueError(f"No Allen stimulus template for frame {frame}")
        image_url = API + by_frame[frame]["download_link"]
        image_path = download(image_url, raw / "images" / f"{frame}.tiff")
        images[str(frame)] = {"url": image_url, "file": str(image_path)}
    digest = hashlib.sha256()
    with nwb.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    manifest = {"session_id": int(session), "nwb_url": url, "nwb_file": str(nwb),
                "nwb_bytes": nwb.stat().st_size, "nwb_sha256": digest.hexdigest(),
                "images": images, "source": "Allen Brain Observatory Visual Coding Neuropixels"}
    (raw / f"session_{session}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=int, default=DEFAULT_SESSION)
    parser.add_argument("--data", default="data")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 30, 60, 90])
    args = parser.parse_args()
    fetch(args.session, args.data, args.frames)
