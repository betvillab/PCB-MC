import argparse
import hashlib
import os
import sys

import requests

API_BASE = "https://data.4tu.nl/v2"
DEFAULT_DOI = "10.4121/7c6e5bda-eecf-4feb-8a3a-7f823b1b0140"


def resolve_article_id(doi: str) -> int:
    resp = requests.post(f"{API_BASE}/articles/search", json={"doi": doi})
    resp.raise_for_status()
    results = resp.json()
    if not results:
        sys.exit(f"No article found for DOI {doi} on {API_BASE}")
    return results[0]["id"]


def list_files(article_id: int) -> list[dict]:
    resp = requests.get(f"{API_BASE}/articles/{article_id}/files")
    resp.raise_for_status()
    return resp.json()


def download_file(file_info: dict, output_dir: str, chunk_size: int = 1 << 20) -> None:
    dest = os.path.join(output_dir, file_info["name"])
    if os.path.exists(dest) and os.path.getsize(dest) == file_info.get("size", -1):
        print(f"skip (already downloaded): {file_info['name']}")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    md5 = hashlib.md5()
    with requests.get(file_info["download_url"], stream=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                md5.update(chunk)

    expected = file_info.get("computed_md5")
    if expected and md5.hexdigest() != expected:
        os.remove(dest)
        sys.exit(f"checksum mismatch for {file_info['name']}: expected {expected}, got {md5.hexdigest()}")

    print(f"downloaded: {file_info['name']} ({file_info.get('size', 0) / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the PCB-MC dataset from 4TU.ResearchData")
    parser.add_argument("--doi", default=DEFAULT_DOI)
    parser.add_argument("--output", default="./data/PCB-MC")
    args = parser.parse_args()

    article_id = resolve_article_id(args.doi)
    files = list_files(article_id)
    os.makedirs(args.output, exist_ok=True)
    for file_info in files:
        download_file(file_info, args.output)


if __name__ == "__main__":
    main()
