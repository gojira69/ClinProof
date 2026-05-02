import sqlite3
import re
import os
import logging

log = logging.getLogger("pubmed_bm25_retriever")

class PubMedBM25Retriever:
    """
    Lightning fast O(1) keyword BM25 retriever for local SQLite FTS5 database.
    """
    def __init__(self, db_path):
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            log.warning(f"FTS5 DB not found at {self.db_path} yet.")
        
    def retrieve(self, query, k=15, **kwargs):
        if not os.path.exists(self.db_path):
            return [], []
            
        # 1. Clean query: only keep alphanumeric to prevent SQLite syntax errors
        clean_query = re.sub(r'[^a-zA-Z0-9 ]', '', str(query))
        
        # Keep sensible word lengths (drop a, I, to, is, etc. trivially if wanted)
        words = [w for w in clean_query.split() if len(w) > 3]
        
        if not words:
            return [], []
            
        # 2. Build FTS5 MATCH string: word1 OR word2 OR word3
        # We can use OR for recall, or AND for precision. OR is safer for BM25 ranking.
        match_str = " OR ".join(words)
        
        docs = []
        scores = []
        
        try:
            # Connect only per query to allow multi-threading in eval
            conn = sqlite3.connect(self.db_path, timeout=10)
            c = conn.cursor()
            
            # 3. Query FTS5 table, order by built-in BM25 rank mechanism (`rank`)
            q = """
                SELECT pmid, abstract, rank
                FROM pubmed_fts 
                WHERE pubmed_fts MATCH ? 
                ORDER BY rank 
                LIMIT ?
            """
            
            c.execute(q, (match_str, k))
            rows = c.fetchall()
            
            for row in rows:
                docs.append({
                    "title": f"PubMed (PMID: {row[0]})",
                    "content": row[1],
                    "source": "pubmed_fts"
                })
                # SQLite FTS rank is usually a negative number
                # More negative = better relevance match
                scores.append(abs(row[2])) 
                
        except Exception as e:
            log.error(f"BM25 Search Error: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
                
        return docs, scores
