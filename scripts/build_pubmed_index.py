"""
ClinProof — Build local PubMed FAISS index from downloaded XML/.xml.gz files.

Run once before using the local retriever:
    python build_pubmed_index.py --xml_dir /path/to/xmls --out_dir /path/to/index

Resumption: safe to kill and restart at any time.
  - Parsed files are cached in {out_dir}/parsed_cache/ (one .json per file)
  - Encoded embedding shards saved in {out_dir}/shards/ (one .npy per 50k abstracts)
  - On restart, already-parsed files and already-encoded shards are skipped
  - Final FAISS merge only runs when all shards are complete

Progress bars:
  - File parsing : tqdm over XML files, shows docs accumulated
  - Encoding     : tqdm per shard with abstracts/sec and ETA
  - FAISS merge  : tqdm over shard files
"""
import xml.etree.ElementTree as ET
import os
import re
import json
import gzip
import logging
import argparse
import glob
import time
import numpy as np
import torch
from xml.etree import ElementTree as ET
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("build_index")

ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
SHARD_SIZE = 50_000   # abstracts per embedding shard (~150MB each at 768d)


# ── XML Parsing ───────────────────────────────────────────────────────────────

def clean(text):
    text = re.sub(r'<[^>]+>', ' ', text or "")
    return re.sub(r'\s+', ' ', text).strip()


def parse_pubmed_xml(xml_path):
    docs = []
    try:
        with gzip.open(xml_path, "rb") as f:
            # We remove the 'tag' argument here for standard library compatibility
            context = ET.iterparse(f, events=("end",))
            try:
                for event, elem in context:
                    # Manually check for the 'PubmedArticle' tag inside the loop
                    if elem.tag != 'PubmedArticle':
                        continue

                    pmid_el = elem.find(".//PMID")
                    if pmid_el is None:
                        elem.clear()
                        continue

                    pmid = pmid_el.text.strip()
                    title_el = elem.find(".//ArticleTitle")
                    title = clean(ET.tostring(
                        title_el, encoding="unicode")) if title_el is not None else ""

                    abs_parts = []
                    for abs_el in elem.findall(".//AbstractText"):
                        label = abs_el.get("Label", "")
                        # Standard ET.tostring behavior can vary; using .text is safer for basic extraction
                        text = clean(" ".join(abs_el.itertext()))
                        abs_parts.append(f"{label}: {text}" if label else text)

                    abstract = " ".join(abs_parts).strip()

                    if abstract:
                        mesh_terms = [
                            mh.find("DescriptorName").text.strip()
                            for mh in elem.findall(".//MeshHeading")
                            if mh.find("DescriptorName") is not None and mh.find("DescriptorName").text
                        ]
                        docs.append({
                            "pmid": pmid,
                            "title": title,
                            "abstract": abstract,
                            "mesh": mesh_terms,
                        })

                    # Memory management
                    elem.clear()
                    # Standard ET doesn't have getprevious(), so we just clear the current element

            except (EOFError, ET.ParseError):
                # This allows us to keep the docs found before the file was cut off
                pass

    except Exception as e:
        log.warning(f"Could not process {xml_path}: {e}")

    return docs

# ── Encoding ──────────────────────────────────────────────────────────────────


def encode_batch(pairs, model):
    with torch.no_grad():
        embs = model.encode(pairs, normalize_embeddings=True,
                            show_progress_bar=False)
    return embs.astype("float32")


# ── Resumption helpers ────────────────────────────────────────────────────────

def get_parse_cache_path(out_dir, xml_path):
    safe = os.path.basename(xml_path).replace(
        ".xml.gz", "").replace(".xml", "")
    return os.path.join(out_dir, "parsed_cache", f"{safe}.json")


def load_progress(out_dir):
    path = os.path.join(out_dir, "build_progress.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"parsed_files": [], "encoded_shards": [], "total_abstracts": 0}


def save_progress(out_dir, state):
    with open(os.path.join(out_dir, "build_progress.json"), "w") as f:
        json.dump(state, f, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def build_index(xml_dir, out_dir, batch_size=64):
    import faiss
    from sentence_transformers import SentenceTransformer

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "parsed_cache"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "shards"),       exist_ok=True)

    progress = load_progress(out_dir)
    already_parsed = set(progress["parsed_files"])
    already_sharded = set(progress["encoded_shards"])

    # ── Step 1: Parse XML files ───────────────────────────────────────────────
    xml_files = sorted(
        glob.glob(os.path.join(xml_dir, "**", "*.xml"),    recursive=True) +
        glob.glob(os.path.join(xml_dir, "**", "*.xml.gz"), recursive=True)
    )

    if not xml_files:
        log.error(f"No XML/.xml.gz files found in {xml_dir}")
        return

    todo_files = [f for f in xml_files if f not in already_parsed]
    log.info(
        f"Found {len(xml_files)} files | "
        f"{len(already_parsed)} already parsed | "
        f"{len(todo_files)} to parse now"
    )

    all_docs = []

    # Load already-parsed docs from per-file cache
    if already_parsed:
        log.info(f"Loading {len(already_parsed)} cached parse results...")
        for xml_path in tqdm(sorted(already_parsed), desc="Loading cache", unit="file", dynamic_ncols=True):
            cache_path = get_parse_cache_path(out_dir, xml_path)
            if os.path.exists(cache_path):
                with open(cache_path) as f:
                    all_docs.extend(json.load(f))
        log.info(f"  Loaded {len(all_docs):,} docs from cache")

    # Parse new files
    if todo_files:
        parse_bar = tqdm(todo_files, desc="Parsing XML",
                         unit="file", dynamic_ncols=True)
        for xml_path in parse_bar:
            parse_bar.set_postfix({"file": os.path.basename(
                xml_path), "total_docs": f"{len(all_docs):,}"})
            docs = parse_pubmed_xml(xml_path)
            if docs:
                cache_path = get_parse_cache_path(out_dir, xml_path)
                with open(cache_path, "w") as f:
                    json.dump(docs, f)
                all_docs.extend(docs)

            already_parsed.add(xml_path)
            progress["parsed_files"] = list(already_parsed)
            progress["total_abstracts"] = len(all_docs)
            save_progress(out_dir, progress)

    if not all_docs:
        log.error("No abstracts found. Check XML format.")
        return

    log.info(f"Total abstracts: {len(all_docs):,}")

    # ── Step 2: Encode in shards ──────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading MedCPT-Article-Encoder on {device}...")
    encoder = SentenceTransformer(ARTICLE_MODEL, device=device)
 
    shard_plan = [
        (shard_id, start, min(start + SHARD_SIZE, len(all_docs)))
        for shard_id, start in enumerate(range(0, len(all_docs), SHARD_SIZE))
    ]
    todo_shards = [(sid, s, e)
                   for sid, s, e in shard_plan if str(sid) not in already_sharded]

    log.info(
        f"Encoding: {len(shard_plan)} shards × up to {SHARD_SIZE:,} abstracts | "
        f"{len(already_sharded)} already done | "
        f"{len(todo_shards)} remaining"
    )

    for shard_id, start, end in todo_shards:
        shard_docs = all_docs[start:end]
        pairs = [[d["title"], d["abstract"]] for d in shard_docs]
        shard_path = os.path.join(
            out_dir, "shards", f"shard_{shard_id:05d}.npy")

        all_embs = []
        t0 = time.time()
        n_encoded = 0

        encode_bar = tqdm(
            range(0, len(pairs), batch_size),
            desc=f"Shard {shard_id+1}/{len(shard_plan)} [{start:,}-{end:,}]",
            unit="batch",
            dynamic_ncols=True,
        )
        for i in encode_bar:
            batch = pairs[i:i + batch_size]
            embs = encode_batch(batch, encoder)
            all_embs.append(embs)
            n_encoded += len(batch)
            elapsed = time.time() - t0
            rate = n_encoded / elapsed if elapsed > 0 else 0
            eta = (len(pairs) - n_encoded) / rate if rate > 0 else 0
            encode_bar.set_postfix({
                "abs/s": f"{rate:.0f}",
                "ETA":   f"{eta/60:.1f}m",
            })

        shard_embs = np.vstack(all_embs).astype("float32")
        np.save(shard_path, shard_embs)
        log.info(
            f"  Saved shard {shard_id} → {shard_embs.shape[0]:,} vectors → {shard_path}")

        already_sharded.add(str(shard_id))
        progress["encoded_shards"] = list(already_sharded)
        save_progress(out_dir, progress)

    # ── Step 3: Merge shards into FAISS ──────────────────────────────────────
    log.info("All shards encoded. Merging into FAISS index...")
    shard_files = sorted(
        glob.glob(os.path.join(out_dir, "shards", "shard_*.npy")))

    first = np.load(shard_files[0])
    dim = first.shape[1]
    index = faiss.IndexFlatIP(dim)

    merge_bar = tqdm(shard_files, desc="Merging shards",
                     unit="shard", dynamic_ncols=True)
    for shard_path in merge_bar:
        embs = np.load(shard_path)
        index.add(embs)
        merge_bar.set_postfix({"vectors": f"{index.ntotal:,}"})

    log.info(f"FAISS index: {index.ntotal:,} vectors, dim={dim}")

    # ── Step 4: Save ─────────────────────────────────────────────────────────
    index_path = os.path.join(out_dir, "pubmed.faiss")
    metadata_path = os.path.join(out_dir, "pubmed_chunks.json")

    faiss.write_index(index, index_path)
    log.info(f"Saved FAISS index → {index_path}")

    meta = [
        {
            "title":   d["title"],
            "content": d["abstract"],
            "pmid":    d["pmid"],
            "mesh":    d["mesh"],
            "source":  "pubmed_local",
        }
        for d in all_docs
    ]
    with open(metadata_path, "w") as f:
        json.dump(meta, f)
    log.info(f"Saved metadata → {metadata_path} ({len(meta):,} records)")

    progress["complete"] = True
    save_progress(out_dir, progress)

    log.info(f"\n{'='*55}")
    log.info(f"  Build complete")
    log.info(f"  Files parsed    : {len(xml_files):,}")
    log.info(f"  Abstracts       : {len(all_docs):,}")
    log.info(f"  Vectors in index: {index.ntotal:,}")
    log.info(f"  FAISS index     : {index_path}")
    log.info(f"  Metadata        : {metadata_path}")
    log.info(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build PubMed FAISS index from local XMLs with resumption support"
    )
    parser.add_argument("--xml_dir",    required=True,
                        help="Directory containing .xml or .xml.gz PubMed files")
    parser.add_argument("--out_dir",    required=True,
                        help="Output directory for index, shards, and metadata")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Encoding batch size (try 128 or 256 on a large GPU)")
    args = parser.parse_args()
    build_index(args.xml_dir, args.out_dir, args.batch_size)
