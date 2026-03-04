"""
ClinProof KG Ingestion: SNOMED CT RF2 Snapshot -> SQLite
"""
import sqlite3, os, sys, logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest_snomed")

def create_schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS snomed_concepts (sctid TEXT PRIMARY KEY, fsn TEXT, pt TEXT, active INTEGER);
    CREATE INDEX IF NOT EXISTS idx_snomed_sctid ON snomed_concepts(sctid);
    CREATE TABLE IF NOT EXISTS snomed_relations (src_sctid TEXT, rel_type TEXT, dst_sctid TEXT, active INTEGER);
    CREATE INDEX IF NOT EXISTS idx_snomed_src ON snomed_relations(src_sctid);
    CREATE TABLE IF NOT EXISTS snomed_to_umls (sctid TEXT, cui TEXT);
    CREATE INDEX IF NOT EXISTS idx_s2u_sctid ON snomed_to_umls(sctid);
    """)
    conn.commit()

def _find_file(directory, prefix):
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.startswith(prefix) and f.endswith(".txt"):
                return os.path.join(root, f)
    return None

def ingest_concepts(snomed_dir, conn):
    con_file = _find_file(snomed_dir, "Concept_Snapshot")
    desc_file = _find_file(snomed_dir, "Description_Snapshot")
    if not con_file or not desc_file:
        log.warning("SNOMED concept/description file not found")
        return
    active = set()
    with open(con_file, encoding="utf-8") as f:
        next(f)
        for line in tqdm(f, desc="concepts"):
            p = line.strip().split("\t")
            if len(p) >= 3 and p[2] == "1": active.add(p[0])
    concept_data = {}
    with open(desc_file, encoding="utf-8") as f:
        next(f)
        for line in tqdm(f, desc="descriptions"):
            p = line.strip().split("\t")
            if len(p) < 9: continue
            active_flag, concept_id, lang, type_id, term = p[2], p[4], p[5], p[6], p[7]
            if active_flag != "1" or lang != "en" or concept_id not in active: continue
            e = concept_data.setdefault(concept_id, {"fsn":"","pt":""})
            if type_id == "900000000000003001": e["fsn"] = term
            elif type_id == "900000000000013009" and not e["pt"]: e["pt"] = term
    rows = [(s, d["fsn"], d["pt"], 1) for s, d in concept_data.items()]
    conn.executemany("INSERT OR IGNORE INTO snomed_concepts VALUES(?,?,?,?)", rows)
    conn.commit()
    log.info(f"SNOMED concepts: {len(rows)}")

def ingest_relationships(snomed_dir, conn):
    rel_file = _find_file(snomed_dir, "Relationship_Snapshot")
    if not rel_file:
        log.warning("SNOMED Relationship file not found"); return
    rows, total = [], 0
    with open(rel_file, encoding="utf-8") as f:
        next(f)
        for line in tqdm(f, desc="relationships"):
            p = line.strip().split("\t")
            if len(p) < 8: continue
            if p[2] != "1": continue
            rows.append((p[4], p[7], p[5], 1))
            if len(rows) >= 100000:
                conn.executemany("INSERT INTO snomed_relations VALUES(?,?,?,?)", rows)
                conn.commit(); total += len(rows); rows = []
    if rows:
        conn.executemany("INSERT INTO snomed_relations VALUES(?,?,?,?)", rows)
        conn.commit(); total += len(rows)
    log.info(f"SNOMED relations: {total}")

def ingest_snomed_to_umls_mapping(umls_meta_dir, conn):
    mrconso = os.path.join(umls_meta_dir, "MRCONSO.RRF")
    if not os.path.exists(mrconso): return
    rows, total = [], 0
    with open(mrconso, encoding="utf-8") as f:
        for line in tqdm(f, desc="SNOMED->UMLS"):
            p = line.strip().split("|")
            if len(p) < 15: continue
            cui, lat, sab, scui = p[0], p[1], p[11], p[9]
            if lat != "ENG" or sab not in ("SNOMEDCT_US","SNOMEDCT") or not scui: continue
            rows.append((scui, cui))
            if len(rows) >= 50000:
                conn.executemany("INSERT OR IGNORE INTO snomed_to_umls VALUES(?,?)", rows)
                conn.commit(); total += len(rows); rows = []
    if rows:
        conn.executemany("INSERT OR IGNORE INTO snomed_to_umls VALUES(?,?)", rows)
        conn.commit(); total += len(rows)
    log.info(f"SNOMED->UMLS: {total}")

def run(snomed_dir, db_path, umls_meta_dir=None):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    create_schema(conn)
    ingest_concepts(snomed_dir, conn)
    ingest_relationships(snomed_dir, conn)
    if umls_meta_dir: ingest_snomed_to_umls_mapping(umls_meta_dir, conn)
    conn.close()
    log.info("SNOMED ingestion complete.")

if __name__ == "__main__":
    import yaml
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    run(cfg["kg"]["snomed_data_path"], cfg["kg"]["db_path"], cfg["kg"]["umls_data_path"])
