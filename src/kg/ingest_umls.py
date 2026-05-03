"""
ClinProof KG Ingestion: UMLS (MRCONSO + MRREL + MRDEF + MRSTY) -> SQLite
Filters English-only, clinically relevant semantic types.
"""
import sqlite3, os, sys, time, logging
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import load_yaml_config, project_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest_umls")

CLINICAL_STYS = {"T047","T048","T184","T033","T046","T191","T121","T109","T195","T126","T116","T023","T022","T029","T060","T061","T059","T201","T034","T200","T203","T074"}
RELEVANT_REL_TYPES = {"RB","RN","SY","CHD","PAR","RO","RL"}
RELEVANT_RELAs = {"has_ingredient","ingredient_of","may_treat","treats","may_prevent","contraindicated_with","cause_of","associated_with","isa","inverse_isa","has_finding_site","part_of","has_part","classified_as"}

def create_schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS umls_concepts (cui TEXT, name TEXT, sty TEXT, sty_name TEXT, lang TEXT DEFAULT 'ENG', is_preferred INTEGER DEFAULT 0, PRIMARY KEY (cui, name));
    CREATE INDEX IF NOT EXISTS idx_concepts_cui ON umls_concepts(cui);
    CREATE INDEX IF NOT EXISTS idx_concepts_name ON umls_concepts(name);
    CREATE TABLE IF NOT EXISTS umls_definitions (cui TEXT, sab TEXT, def TEXT, PRIMARY KEY (cui, sab));
    CREATE INDEX IF NOT EXISTS idx_defs_cui ON umls_definitions(cui);
    CREATE TABLE IF NOT EXISTS umls_relations (cui1 TEXT, rel TEXT, rela TEXT, cui2 TEXT, sab TEXT);
    CREATE INDEX IF NOT EXISTS idx_rel_cui1 ON umls_relations(cui1);
    CREATE INDEX IF NOT EXISTS idx_rel_cui2 ON umls_relations(cui2);
    CREATE TABLE IF NOT EXISTS umls_stys (cui TEXT, sty TEXT, sty_abbr TEXT, PRIMARY KEY (cui, sty));
    CREATE INDEX IF NOT EXISTS idx_stys_cui ON umls_stys(cui);
    """)
    conn.commit()

def ingest_mrsty(meta_dir, conn):
    path = os.path.join(meta_dir, "MRSTY.RRF")
    clinical_cuis, rows = set(), []
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="MRSTY"):
            p = line.strip().split("|")
            if len(p) < 4: continue
            rows.append((p[0], p[3], p[1]))
            if p[1] in CLINICAL_STYS: clinical_cuis.add(p[0])
    conn.executemany("INSERT OR IGNORE INTO umls_stys VALUES (?,?,?)", rows)
    conn.commit()
    log.info(f"MRSTY: {len(rows)} rows, {len(clinical_cuis)} clinical CUIs")
    return clinical_cuis

def ingest_mrconso(meta_dir, conn, clinical_cuis):
    path = os.path.join(meta_dir, "MRCONSO.RRF")
    rows, total = [], 0
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="MRCONSO"):
            p = line.strip().split("|")
            if len(p) < 15: continue
            cui, lat, ts, _, stt, _, ispref = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            str_ = p[14]
            if lat != "ENG" or cui not in clinical_cuis: continue
            is_pref = 1 if (ispref == "Y" and ts == "P") else 0
            rows.append((cui, str_, lat, is_pref))
            if len(rows) >= 50000:
                conn.executemany("INSERT OR IGNORE INTO umls_concepts(cui,name,lang,is_preferred) VALUES(?,?,?,?)", rows)
                conn.commit(); total += len(rows); rows = []
    if rows:
        conn.executemany("INSERT OR IGNORE INTO umls_concepts(cui,name,lang,is_preferred) VALUES(?,?,?,?)", rows)
        conn.commit(); total += len(rows)
    log.info(f"MRCONSO: {total} entries")

def ingest_mrdef(meta_dir, conn, clinical_cuis):
    path = os.path.join(meta_dir, "MRDEF.RRF")
    rows, total = [], 0
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="MRDEF"):
            p = line.strip().split("|")
            if len(p) < 6: continue
            cui, sab, def_ = p[0], p[4], p[5]
            if cui not in clinical_cuis: continue
            rows.append((cui, sab, def_))
            if len(rows) >= 10000:
                conn.executemany("INSERT OR IGNORE INTO umls_definitions VALUES(?,?,?)", rows)
                conn.commit(); total += len(rows); rows = []
    if rows:
        conn.executemany("INSERT OR IGNORE INTO umls_definitions VALUES(?,?,?)", rows)
        conn.commit(); total += len(rows)
    log.info(f"MRDEF: {total} definitions")

def ingest_mrrel(meta_dir, conn, clinical_cuis):
    path = os.path.join(meta_dir, "MRREL.RRF")
    rows, total = [], 0
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="MRREL"):
            p = line.strip().split("|")
            if len(p) < 11: continue
            cui1, rel, cui2, rela, sab = p[0], p[3], p[4], p[7], p[10]
            if cui1 not in clinical_cuis or cui2 not in clinical_cuis: continue
            if rel not in RELEVANT_REL_TYPES and rela.lower() not in RELEVANT_RELAs: continue
            rows.append((cui1, rel, rela, cui2, sab))
            if len(rows) >= 100000:
                conn.executemany("INSERT INTO umls_relations VALUES(?,?,?,?,?)", rows)
                conn.commit(); total += len(rows); rows = []
    if rows:
        conn.executemany("INSERT INTO umls_relations VALUES(?,?,?,?,?)", rows)
        conn.commit(); total += len(rows)
    log.info(f"MRREL: {total} relations")

def run(meta_dir, db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-500000")
    create_schema(conn)
    cuis = ingest_mrsty(meta_dir, conn)
    ingest_mrconso(meta_dir, conn, cuis)
    ingest_mrdef(meta_dir, conn, cuis)
    ingest_mrrel(meta_dir, conn, cuis)
    conn.close()
    log.info("UMLS ingestion complete.")

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else project_path("config", "default.yaml")
    cfg = load_yaml_config(cfg_path)
    run(cfg["kg"]["umls_data_path"], cfg["kg"]["db_path"])
