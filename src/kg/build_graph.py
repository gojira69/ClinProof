"""
ClinProof KG Builder: SQLite -> NetworkX DiGraph
Merges UMLS, SNOMED CT, and RxNorm into a unified typed knowledge graph.
"""
import sqlite3, pickle, os, sys, logging
import networkx as nx
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import load_yaml_config, project_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_graph")

SNOMED_TYPE_LABELS = {"116680003":"isa","363698007":"finding_site","246075003":"causative_agent","47429007":"associated_with"}

def build_graph(db_path, graph_path):
    log.info("Building KG from SQLite...")
    conn = sqlite3.connect(db_path)
    G = nx.DiGraph()

    # UMLS nodes
    cur = conn.execute("SELECT c.cui, c.name, s.sty, d.def FROM umls_concepts c LEFT JOIN umls_stys s ON c.cui=s.cui LEFT JOIN umls_definitions d ON c.cui=d.cui WHERE c.is_preferred=1 AND c.lang='ENG'")
    seen = set()
    for cui, name, sty, defn in tqdm(cur, desc="UMLS nodes"):
        if cui in seen: continue
        seen.add(cui)
        G.add_node(cui, label=name or cui, node_type="umls", sty=sty or "", definition=(defn or "")[:400], source="umls")
    log.info(f"UMLS nodes: {len(seen)}")

    # UMLS edges
    cur = conn.execute("SELECT cui1, rel, rela, cui2, sab FROM umls_relations WHERE cui1 IN (SELECT DISTINCT cui FROM umls_concepts) AND cui2 IN (SELECT DISTINCT cui FROM umls_concepts)")
    ec = 0
    for cui1, rel, rela, cui2, sab in tqdm(cur, desc="UMLS edges"):
        if G.has_node(cui1) and G.has_node(cui2):
            G.add_edge(cui1, cui2, rel=rela or rel, source="umls"); ec += 1
    log.info(f"UMLS edges: {ec}")

    # SNOMED->UMLS map
    snomed_to_cui = {}
    try:
        for row in conn.execute("SELECT sctid, cui FROM snomed_to_umls"):
            snomed_to_cui[row[0]] = row[1]
    except Exception: pass

    # SNOMED concept nodes (enrich existing or add new)
    try:
        cur = conn.execute("SELECT sctid, fsn, pt FROM snomed_concepts WHERE active=1")
        sct_count = 0
        for sctid, fsn, pt in tqdm(cur, desc="SNOMED nodes"):
            cui = snomed_to_cui.get(sctid)
            node_id = cui if (cui and G.has_node(cui)) else f"SCT:{sctid}"
            if node_id in G:
                if pt: G.nodes[node_id]["snomed_pt"] = pt
                G.nodes[node_id]["sctid"] = sctid
            else:
                G.add_node(node_id, label=pt or fsn or sctid, node_type="snomed", sctid=sctid, source="snomed")
            sct_count += 1
        log.info(f"SNOMED concepts: {sct_count}")

        cur = conn.execute("SELECT src_sctid, rel_type, dst_sctid FROM snomed_relations WHERE active=1")
        sec = 0
        for src, rel_type, dst in tqdm(cur, desc="SNOMED edges"):
            s_node = snomed_to_cui.get(src, f"SCT:{src}")
            d_node = snomed_to_cui.get(dst, f"SCT:{dst}")
            if G.has_node(s_node) and G.has_node(d_node):
                G.add_edge(s_node, d_node, rel=SNOMED_TYPE_LABELS.get(rel_type, f"sct_{rel_type}"), source="snomed"); sec += 1
        log.info(f"SNOMED edges: {sec}")
    except Exception as e:
        log.warning(f"SNOMED data not found or error: {e}")

    # RxNorm drug nodes
    try:
        cur = conn.execute("SELECT rxcui, name, tty FROM rxnorm_drugs WHERE tty IN ('IN','BN','PIN','MIN') GROUP BY rxcui")
        rx_count = 0
        for rxcui, name, tty in tqdm(cur, desc="RxNorm nodes"):
            node_id = f"RX:{rxcui}"
            G.add_node(node_id, label=name, node_type="rxnorm", rxcui=rxcui, tty=tty, source="rxnorm"); rx_count += 1
        log.info(f"RxNorm nodes: {rx_count}")

        cur = conn.execute("SELECT rxcui1, rela, rxcui2 FROM rxnorm_relations")
        rxec = 0
        for rxcui1, rela, rxcui2 in tqdm(cur, desc="RxNorm edges"):
            s, d = f"RX:{rxcui1}", f"RX:{rxcui2}"
            if G.has_node(s) and G.has_node(d):
                G.add_edge(s, d, rel=rela or "related", source="rxnorm"); rxec += 1
        log.info(f"RxNorm edges: {rxec}")
    except Exception as e:
        log.warning(f"RxNorm data not found or error: {e}")

    conn.close()
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    log.info(f"Saving KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges -> {graph_path}")
    with open(graph_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    log.info("KG saved.")
    return G

def load_graph(graph_path):
    with open(graph_path, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else project_path("config", "default.yaml")
    cfg = load_yaml_config(cfg_path)
    build_graph(cfg["kg"]["db_path"], cfg["kg"]["graph_path"])
