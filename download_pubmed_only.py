import sys, os
import logging

# Ensure MedRAG is in the path
sys.path.insert(0, "/mnt/d/Harsha/AoLM/project/MedRAG/src")

try:
    from medrag import MedRAG
except ImportError:
    print("Error: Could not find MedRAG source. Please check the path in sys.path.insert.")
    sys.exit(1)

logging.getLogger('httpx').setLevel(logging.WARNING)
os.environ["OPENAI_API_KEY"] = "dummy" # Not needed for download

def main():
    print("="*60)
    print("  Initializing PubMed Corpus (Abstracts + Embeddings)")
    print("  Source: MedRAG / MedCite Version")
    print("  This will download ~25-30GB of data via git/wget.")
    print("="*60)
    
    # Setting corpus_name to "PubMed" specifically
    # Using the local path for the database
    medrag = MedRAG(
        llm_name="OpenAI/gpt-3.5-turbo-16k", 
        rag=True, 
        retriever_name="MedCPT", 
        corpus_name="PubMed", 
        db_dir="/mnt/d/Harsha/AoLM/project/MedRAG/corpus",
        corpus_cache=True 
    )

    print("\n✅ PubMed Download and Indexing complete!")
    print("Location: d:\\Harsha\\AoLM\\project\\MedRAG\\corpus\\PubMed")

if __name__ == "__main__":
    main()
