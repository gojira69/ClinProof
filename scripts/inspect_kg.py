import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import project_path

print("Loading KG...")
with open(project_path("data", "kg_graph.pkl"), "rb") as f:
    G = pickle.load(f)

carvedilol_nodes = [n for n, d in G.nodes(data=True) if str(d.get('label')).lower() == 'carvedilol']
hf_nodes = [n for n, d in G.nodes(data=True) if str(d.get('label')).lower() == 'heart failure, congestive' or str(d.get('label')).lower() == 'heart failure']

for node in carvedilol_nodes:
    print(f"\nCarvedilol relations for {node}:")
    for _, nbr, ed in G.out_edges(node, data=True):
        print(f"  -> {ed.get('rel')} -> {G.nodes[nbr].get('label', nbr)}")
        
for node in hf_nodes:
    print(f"\nHeart failure relations for {node}:")
    for _, nbr, ed in G.out_edges(node, data=True):
        print(f"  -> {ed.get('rel')} -> {G.nodes[nbr].get('label', nbr)}")
