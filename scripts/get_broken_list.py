import os
import gzip
import zlib
import logging
from tqdm import tqdm
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import project_path

# Update this path if your directory structure has changed
XML_DIR = project_path("pubmed_download", "pubmed_abstracts")

def verify_stream_integrity(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} not found.")
        return

    files = sorted([f for f in os.listdir(directory) if f.endswith(".xml.gz")])
    broken = []

    print(f"Testing {len(files)} files for stream integrity...")
    # Using a standard loop to catch the specific exception and continue
    for filename in tqdm(files):
        path = os.path.join(directory, filename)
        try:
            with gzip.open(path, 'rb') as f:
                # Attempting to read a small chunk from the end 
                # forces gzip to verify the checksum and trailer
                f.seek(-1, os.SEEK_END)
                f.read(1)
        except (EOFError, OSError, zlib.error):
            broken.append(path)
            
    if broken:
        print(f"\n[!] Detected {len(broken)} truncated or corrupted files.")
        for b in broken:
            print(f"Deleting: {os.path.basename(b)}")
            os.remove(b)
        print("\nBroken files removed. Restart your downloader to fetch clean copies.")
    else:
        print("\nAll files are healthy and ready for indexing.")

if __name__ == "__main__":
    verify_stream_integrity(XML_DIR)
