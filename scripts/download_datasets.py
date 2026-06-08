"""Download the ASCAD dataset required for the sanity check and experiments.

Usage:
    python scripts/download_datasets.py                     # download to ~/Downloads (default)
    python scripts/download_datasets.py --dest /data/path   # custom destination

Expected layout after download:
    ~/Downloads/ASCAD/ASCAD_desync0/ASCAD_desync0.h5
"""

import argparse
import os
import sys

try:
    import gdown
except ImportError:
    print("gdown is required. Install it with: pip install gdown")
    sys.exit(1)


# Google Drive folder containing organised datasets
FOLDER_ID = "1GcWQvwwEdbj2L0c1hd2YpLpbS-gIFJJ5"

# Individual fallback URLs documented in the README
DATASET_URLS = {
    "ASCAD_desync0": (
        "https://github.com/ANSSI-FR/ASCAD/raw/master/"
        "ATMEGA_AES_v1/ATM_AES_v1_fixed_key/ASCAD_data/ASCAD_databases/ASCAD.h5"
    ),
}


def download_folder(dest: str) -> bool:
    print(f"Downloading dataset folder from Google Drive (ID: {FOLDER_ID}) ...")
    print(f"Destination: {dest}")
    try:
        gdown.download_folder(id=FOLDER_ID, output=dest, quiet=False)
        return True
    except Exception as e:
        print(f"Folder download failed: {e}")
        return False


def download_fallback(dest: str) -> None:
    print("Falling back to individual downloads ...\n")
    for name, url in DATASET_URLS.items():
        out_dir = os.path.join(dest, "ASCAD", name)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{name}.h5")
        if os.path.isfile(out_file):
            print(f"[SKIP] {out_file} already exists")
            continue
        print(f"Downloading {name} -> {out_file}")
        try:
            gdown.download(url, output=out_file, quiet=False)
        except Exception as e:
            print(f"  Failed: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ASCAD / SCA datasets for deepscapy experiments"
    )
    parser.add_argument(
        "--dest",
        default=os.path.join(os.path.expanduser("~"), "Downloads"),
        help="Destination directory (default: ~/Downloads)",
    )
    args = parser.parse_args()

    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)

    if not download_folder(dest):
        download_fallback(dest)

    expected = os.path.join(dest, "ASCAD", "ASCAD_desync0", "ASCAD_desync0.h5")
    if os.path.isfile(expected):
        print(f"\nSUCCESS: dataset ready at {expected}")
    else:
        print(
            f"\nNOTE: {expected} not found. "
            "If the download succeeded but the layout differs, adjust manually."
        )


if __name__ == "__main__":
    main()
