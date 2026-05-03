import sys
import os
import json
from pathlib import Path

# Add the project root to the python path so imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.retrieval.pubmed_api_retriever import PubMedAPIRetriever
from src.utils.paths import project_path
import logging

logging.basicConfig(level=logging.INFO)

def test_pubmed_api():
    config = {
        "pubmed": {
            "email": "test@example.com",
            "api_key": "",
            "max_results": 10, # How many articles to fetch from NCBI Entrez
            "cache_dir": project_path("data", "pubmed_cache")
        }
    }
    
    print("--- Initializing PubMed API Retriever ---")
    retriever = PubMedAPIRetriever(config)
    
    query = "Is the protein Papilin secreted?"
    print(f"\n--- Running query: '{query}' ---")
    
    # Run the retrieval. k=5 means we want the top 5 most relevant semantic chunks returned
    # pubmed_mode="pubmed" means we'll query the PubMed (abstracts) database.
    # If pubmed_mode="pmc", it searches PMC full texts first, then falls back to pubmed.
    chunks, scores = retriever.retrieve(query, k=5, pubmed_mode="pubmed")
    
    print(f"\n--- Retrieved {len(chunks)} top semantic chunks ---")
    for i, (chunk, score) in enumerate(zip(chunks, scores)):
        print(f"\n[Rank {i+1} | Score: {score:.4f}]")
        print(f"Title: {chunk['title']}")
        print(f"PMID: {chunk.get('PMID', 'N/A')}")
        print(f"Source DB: {chunk['source']}")
        print(f"Content snippet (first 150 chars): {chunk['content'][:150]}...")
        
    # Also, we can inspect exactly what a raw chunk dictionary looks like
    if chunks:
        print("\n--- Example Raw Chunk Dictionary ---")
        print(json.dumps(chunks[0], indent=2))

if __name__ == "__main__":
    test_pubmed_api()
