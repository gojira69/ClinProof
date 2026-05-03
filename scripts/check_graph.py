from collections import Counter
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import project_path

with open(project_path("data", "kg_graph.pkl"), "rb") as handle:
    G = pickle.load(handle)

# What semantic types exist?
sty_counts = Counter(d.get("sty","unknown") for _, d in G.nodes(data=True))
for sty, count in sty_counts.most_common(30):
    print(f"{count:>8}  {sty}")

# What node types?
type_counts = Counter(d.get("node_type","unknown") for _, d in G.nodes(data=True))
print(type_counts)

# How many nodes have definitions?
has_def = sum(1 for _, d in G.nodes(data=True) if d.get("definition","").strip())
print(f"{has_def}/{G.number_of_nodes()} nodes have definitions")
