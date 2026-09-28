"""Download and verify the original paper's text-to-image checkpoint."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time

from models.config import (
    MANIFEST_PATH,
    MODEL_DIR,
    MODEL_FILES,
    MODEL_ID,
    REVISION,
    ROOT,
)

BLOCK_SIZE = 16 * 1024**2


def reuse_chunk_prefix(path, entry):
    """Extend a resumable file using contiguous complete legacy range blocks."""
    partial = path.with_name(path.name + ".part")
    directory = path.with_name(path.name + ".parts")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > entry["size"]:
        raise RuntimeError(f"Oversized partial file: {partial}. Move it aside and retry.")
    initial = offset
    while offset < entry["size"] and directory.is_dir():
        start = offset // BLOCK_SIZE * BLOCK_SIZE
        size = min(BLOCK_SIZE, entry["size"] - start)
        chunk = directory / f"{start:012d}"
        if not chunk.is_file() or chunk.stat().st_size != size:
            break
        with chunk.open("rb") as source, partial.open("ab") as output:
            source.seek(offset - start)
            shutil.copyfileobj(source, output, length=8 * 1024**2)
        offset = start + size
    if offset > initial:
        print(f"Reused blocks: {entry['rfilename']} ({(offset - initial) / 1024**2:.1f} MiB)", flush=True)
    return partial


def verify_file(path, entry):
    if not path.is_file() or path.stat().st_size != entry["size"]:
        return False
    lfs = entry.get("lfs")
    digest = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        digest.update(f"blob {entry['size']}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == (lfs["sha256"] if lfs else entry["blobId"])


def curl_download(entry):
    path = MODEL_DIR / entry["rfilename"]
    if verify_file(path, entry):
        print(f"Verified: {entry['rfilename']}", flush=True)
        return
    if path.exists():
        raise RuntimeError(f"Existing file failed verification: {path}. Move it aside and retry.")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = reuse_chunk_prefix(path, entry)
    if verify_file(partial, entry):
        partial.replace(path)
        print(f"Verified: {entry['rfilename']}", flush=True)
        return
    if partial.exists() and partial.stat().st_size == entry["size"]:
        raise RuntimeError(f"Size/hash mismatch: {partial}. Move it aside and retry.")
    url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{entry['rfilename']}?download=true"
    print(f"Downloading: {entry['rfilename']} ({entry['size'] / 1024**2:.1f} MiB)", flush=True)
    process = subprocess.Popen([
        "curl", "--location", "--fail", "--silent", "--show-error",
        "--connect-timeout", "30", "--speed-limit", "1024", "--speed-time", "120", "--retry", "4",
        "--retry-all-errors", "--retry-delay", "2", "--continue-at", "-",
        "--output", str(partial), url,
    ])
    last_report = time.monotonic()
    last_size = partial.stat().st_size if partial.exists() else 0
    try:
        while process.poll() is None:
            time.sleep(1)
            now = time.monotonic()
            if now - last_report >= 5:
                size = partial.stat().st_size if partial.exists() else 0
                speed = max(0, size - last_size) / (now - last_report) / 1024**2
                print(f"  {entry['rfilename']}: {size / entry['size']:.1%}, {speed:.1f} MiB/s", flush=True)
                last_report, last_size = now, size
    except BaseException:
        process.terminate()
        process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f"Download failed: {path.name}; rerun to resume {partial.name}.")
    if not verify_file(partial, entry):
        raise RuntimeError(f"Size/hash mismatch: {partial}. File was NOT installed as a model.")
    partial.replace(path)
    print(f"Verified: {entry['rfilename']}", flush=True)


def chunked_download(entries, workers):
    """Bounded byte ranges survive proxies that terminate long HTTP streams."""
    block_size = BLOCK_SIZE
    jobs = []
    pending = []
    cancelled = threading.Event()
    for entry in entries:
        target = MODEL_DIR / entry["rfilename"]
        if verify_file(target, entry):
            print(f"Verified: {entry['rfilename']}", flush=True)
            continue
        if target.exists():
            raise RuntimeError(f"Existing file failed verification: {target}")
        if entry["size"] < block_size:
            curl_download(entry)
            continue
        directory = target.with_name(target.name + ".parts")
        directory.mkdir(parents=True, exist_ok=True)
        pending.append((entry, target, directory))
        # Reuse complete blocks from an earlier uninterrupted prefix download.
        prefix = target.with_name(target.name + ".part")
        for start in range(0, entry["size"], block_size):
            end = min(start + block_size, entry["size"]) - 1
            chunk = directory / f"{start:012d}"
            if not chunk.exists() and prefix.exists() and prefix.stat().st_size >= end + 1:
                with prefix.open("rb") as stream:
                    stream.seek(start)
                    chunk.write_bytes(stream.read(end - start + 1))
            jobs.append((entry, start, end, chunk))

    def fetch(job):
        entry, start, end, chunk = job
        size = end - start + 1
        if chunk.exists() and chunk.stat().st_size == size:
            return size
        temporary = chunk.with_suffix(".tmp")
        headers = chunk.with_suffix(".headers")
        for attempt in range(20):
            if cancelled.is_set():
                raise RuntimeError("Download cancelled; partial blocks retained.")
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset == size:
                break
            if offset > size:
                raise RuntimeError(f"Oversized partial block: {temporary}")
            range_start = start + offset
            tail = chunk.with_suffix(".tail")
            tail.unlink(missing_ok=True)
            url = (f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{entry['rfilename']}"
                   f"?download=true&range_start={range_start}&range_end={end}")
            process = subprocess.Popen([
                "curl", "--http1.1", "--location", "--fail", "--silent", "--show-error",
                "--connect-timeout", "20", "--max-time", "120", "--range", f"{range_start}-{end}",
                "--max-filesize", str(size - offset), "--dump-header", str(headers),
                "--output", str(tail), url,
            ], stderr=subprocess.DEVNULL)
            while process.poll() is None:
                if cancelled.wait(0.5):
                    process.terminate()
                    process.wait()
                    raise RuntimeError("Download cancelled; partial blocks retained.")
            # Even on timeout, a valid HTTP 206 body is a reusable prefix. Check
            # Content-Range before appending; never append an error page.
            ranges = re.findall(r"content-range:\s*bytes (\d+)-(\d+)/(\d+)",
                                headers.read_text().lower() if headers.exists() else "")
            expected_range = (str(range_start), str(end), str(entry["size"]))
            if ranges and ranges[-1] == expected_range and tail.exists() and 0 < tail.stat().st_size <= size - offset:
                with temporary.open("ab") as output, tail.open("rb") as source:
                    shutil.copyfileobj(source, output)
                tail.unlink()
            if temporary.exists() and temporary.stat().st_size == size:
                break
            cancelled.wait(1)
        else:
            raise RuntimeError(f"Repeated network errors: {entry['rfilename']} block {start}; rerun to resume.")
        headers.unlink(missing_ok=True)
        if temporary.stat().st_size != size:
            raise RuntimeError(f"Incorrect range length: {entry['rfilename']} {start}-{end}")
        temporary.replace(chunk)
        return size

    print(f"Downloading {len(jobs)} blocks with {workers} connections; completed blocks are reusable.", flush=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, job) for job in jobs]
        try:
            for index, future in enumerate(as_completed(futures), 1):
                completed += future.result()
                print(f"Blocks: {index}/{len(jobs)}, {completed / 1024**3:.2f} GiB verified by length", flush=True)
        except BaseException:
            cancelled.set()
            for future in futures:
                future.cancel()
            raise
    for entry, target, directory in pending:
        assembled = target.with_name(target.name + ".assembled")
        with assembled.open("wb") as output:
            for start in range(0, entry["size"], block_size):
                with (directory / f"{start:012d}").open("rb") as stream:
                    shutil.copyfileobj(stream, output)
        if not verify_file(assembled, entry):
            raise RuntimeError(f"SHA256 mismatch: {assembled}; NOT installed as a model.")
        assembled.replace(target)
        print(f"SHA256 verified: {entry['rfilename']}", flush=True)
        # Delete only this downloader's temporary blocks, after full validation.
        for chunk in directory.iterdir():
            chunk.unlink()
        directory.rmdir()
        target.with_name(target.name + ".part").unlink(missing_ok=True)


def configure_download_environment():
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def main():
    configure_download_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=["chunks", "curl", "hub"], default="curl",
                        help="curl: continuous files (default); chunks: unreliable proxies; hub: Hugging Face")
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=8,
                        help="Concurrent files (curl/hub) or byte ranges (chunks), default: 8")
    args = parser.parse_args()
    if args.transport != "hub" and not shutil.which("curl"):
        parser.error("curl is missing; use --transport hub.")

    free = shutil.disk_usage(ROOT).free / 1024**3
    print(f"Model: {MODEL_ID}\nRevision: {REVISION}\nDestination: {MODEL_DIR}", flush=True)
    print(f"Weights: approximately 5.73 GiB; free disk: {free:.1f} GiB", flush=True)
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest["revision"] != REVISION or manifest["model_id"] != MODEL_ID:
        raise RuntimeError("Model manifest does not match the pinned model.")
    remaining = sum(entry["size"] for entry in manifest["files"]
                    if not (MODEL_DIR / entry["rfilename"]).is_file()) / 1024**3
    required = (13 if args.transport == "chunks" else remaining + 1) if remaining else 0
    if free < required:
        raise SystemExit(f"Please free at least {required:.1f} GiB before downloading the weights.")
    print(f"Transport: {args.transport}; workers: {args.workers}", flush=True)
    if args.transport == "chunks":
        chunked_download(manifest["files"], args.workers)
    elif args.transport == "curl":
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(curl_download, manifest["files"]))
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=MODEL_ID, revision=REVISION, local_dir=MODEL_DIR,
            allow_patterns=MODEL_FILES, max_workers=args.workers,
        )
        for entry in manifest["files"]:
            if not verify_file(MODEL_DIR / entry["rfilename"], entry):
                raise RuntimeError(f"Size/hash mismatch: {entry['rfilename']}")
    print("Download complete. Run: python generate.py --prompt 'A red fox in a snowy forest, oil painting'", flush=True)


if __name__ == "__main__":
    main()
