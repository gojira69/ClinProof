import sys, os
import logging
from pathlib import Path

# Add MedRAG to path
PROJECT_ROOT = Path(__file__).resolve().parent
MEDRAG_SRC = Path(os.environ.get("MEDRAG_SRC_DIR", PROJECT_ROOT / "external" / "MedRAG" / "src"))
MEDRAG_CORPUS_DIR = Path(os.environ.get("MEDRAG_CORPUS_DIR", PROJECT_ROOT / "data" / "medrag_corpus"))
sys.path.insert(0, str(MEDRAG_SRC))
from medrag import MedRAG

logging.getLogger('httpx').setLevel(logging.WARNING)
os.environ["OPENAI_API_KEY"] = "dummy"

def main():
    print("="*60)
    print("  Initializing MedRAG Corpora (MedCorp + MedCPT)")
    print("  This will trigger `wget` and `git clone` commands")
    print("  to pull ~30GB of text and dense Faiss embeddings.")
    print("="*60)
    
    # Initializing this object with rag=True automatically runs all checks
    # to download missing chunks and embeddings. As MedRAG uses standard
    # stdout for git and wget, you will see native progress bars here!
    medrag = MedRAG(
        llm_name="OpenAI/gpt-3.5-turbo-16k", 
        rag=True, 
        retriever_name="MedCPT", 
        corpus_name="MedCorp", 
        db_dir=str(MEDRAG_CORPUS_DIR),
        corpus_cache=True # Also builds the cache index (id2text.json)
    )

    print("\n✅ Download and indexing complete!")
    print("You can now run `eval_medrag.py` and `eval_medcite.py` without delays.")

if __name__ == "__main__":
    main()
