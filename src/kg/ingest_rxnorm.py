"""
ClinProof KG Ingestion: RxNorm RRF -> SQLite
"""
import sqlite3, os, sys, logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ingest_rxnorm")

DRUG_RELS = {"has_ingredient","ingredient_of","has_tradename","tradename_of","contains","contained_in","isa","inverse_isa"}
DRUG_ATTRS = {"RXAUI_STRENGTH","RXN_STRENGTH","RXN_HUMAN_DRUG","NDC","UMLSCUI"}
USEFUL_TTY = {"IN","BN","PIN","MIN","SCDF","SCD","BPCK","GPCK","SBD","SBDF"}

def create_schema(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS rxnorm_drugs (rxcui TEXT, name TEXT, tty TEXT, is_active INTEGER DEFAULT 1, PRIMARY KEY (rxcui, name, tty));
    CREATE INDEX IF NOT EXISTS idx_rx_rxcui ON rxnorm_drugs(rxcui);
    CREATE INDEX IF NOT EXISTS idx_rx_name ON rxnorm_drugs(name);
    CREATE TABLE IF NOT EXISTS rxnorm_relations (rxcui1 TEXT, rel TEXT, rela TEXT, rxcui2 TEXT, sab TEXT);
    CREATE INDEX IF NOT EXISTS idx_rxrel_cui1 ON rxnorm_relations(rxcui1);
    CREATE TABLE IF NOT EXISTS rxnorm_attributes (rxcui TEXT, atn TEXT, atv TEXT);
    CREATE INDEX IF NOT EXISTS idx_rxattr_cui ON rxnorm_attributes(rxcui);
    """)
    conn.commit()

def ingest_rxnconso(rrf_dir, conn):
    path = os.path.join(rrf_dir, "RXNCONSO.RRF")
    rows, total, drug_rxcuis = [], 0, set()
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="RXNCONSO"):
            p = line.strip().split("|")
            if len(p) < 15: continue
            rxcui, lat, tty, sab = p[0], p[1], p[12], p[11]
            name = p[14] if len(p) > 14 else p[13]
            if (lat != "ENG" and sab != "RXNORM") or tty not in USEFUL_TTY: continue
            drug_rxcuis.add(rxcui)
            rows.append((rxcui, name, tty, 1))
            if len(rows) >= 50000:
                conn.executemany("INSERT OR IGNORE INTO rxnorm_drugs VALUES(?,?,?,?)", rows)
                conn.commit(); total += len(rows); rows = []
    if rows:
        conn.executemany("INSERT OR IGNORE INTO rxnorm_drugs VALUES(?,?,?,?)", rows)
        conn.commit(); total += len(rows)
    log.info(f"RXNCONSO: {total}, {len(drug_rxcuis)} RXCUIs")
    return drug_rxcuis

def ingest_rxnrel(rrf_dir, conn, drug_rxcuis):
    path = os.path.join(rrf_dir, "RXNREL.RRF")
    rows, total = [], 0
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="RXNREL"):
            p = line.strip().split("|")
            if len(p) < 11: continue
            rxcui1, rel, rxcui2, rela, sab = p[0], p[3], p[4], p[7].lower() if p[7] else "", p[10]
            if rxcui1 not in drug_rxcuis or rxcui2 not in drug_rxcuis: continue
            if rela and rela not in DRUG_RELS: continue
            rows.append((rxcui1, rel, rela, rxcui2, sab))
            if len(rows) >= 100000:
                conn.executemany("INSERT INTO rxnorm_relations VALUES(?,?,?,?,?)", rows)
                conn.commit(); total += len(rows); rows = []
    if rows:
        conn.executemany("INSERT INTO rxnorm_relations VALUES(?,?,?,?,?)", rows)
        conn.commit(); total += len(rows)
    log.info(f"RXNREL: {total}")

def ingest_rxnsat(rrf_dir, conn, drug_rxcuis):
    path = os.path.join(rrf_dir, "RXNSAT.RRF")
    rows, total = [], 0
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="RXNSAT"):
            p = line.strip().split("|")
            if len(p) < 11: continue
            rxcui, atn, atv = p[0], p[8], p[10]
            if rxcui not in drug_rxcuis or atn not in DRUG_ATTRS: continue
            rows.append((rxcui, atn, atv))
            if len(rows) >= 50000:
                conn.executemany("INSERT INTO rxnorm_attributes VALUES(?,?,?)", rows)
                conn.commit(); total += len(rows); rows = []
    if rows:
        conn.executemany("INSERT INTO rxnorm_attributes VALUES(?,?,?)", rows)
        conn.commit(); total += len(rows)
    log.info(f"RXNSAT: {total}")

def run(rrf_dir, db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-200000")
    create_schema(conn)
    drug_rxcuis = ingest_rxnconso(rrf_dir, conn)
    ingest_rxnrel(rrf_dir, conn, drug_rxcuis)
    ingest_rxnsat(rrf_dir, conn, drug_rxcuis)
    conn.close()
    log.info("RxNorm ingestion complete.")

if __name__ == "__main__":
    import yaml
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    run(cfg["kg"]["rxnorm_data_path"], cfg["kg"]["db_path"])
