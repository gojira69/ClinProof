import sys, os, json, time, yaml, traceback
import pandas as pd
import argparse

# Add MedRAG to path
sys.path.insert(0, "/mnt/d/Harsha/AoLM/project/MedRAG/src")
from medrag import MedRAG

# Add ClinProof to path
sys.path.insert(0, "/mnt/d/Harsha/AoLM/project/clinproof")
from src.generation.ollama_llm import OllamaLLM
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.moe_retriever import MoERetriever
from src.retrieval.pubmed_dense_retriever import PubMedDenseRetriever

CONFIG = {
    "count": 100,
    "model": "llama3.1:8b",
    "resume": True,
    "tag": "medrag_clinproof_retrieval",
    "k": 25,
    "context_len": 8000,
    "use_pubmed": True,
    "use_graph": False
}

def load_dataset(name, count):
    if name == "BioASQ":
        with open("/mnt/d/Harsha/AoLM/project/data/BioASQ-training13b/training13b.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        qs = [{"id": q.get("id", str(i)), "question": q["body"], "options": {"A": "Yes", "B": "No"}, 
               "answer": "A" if q.get("exact_answer", "").lower() == "yes" else "B"} 
              for i, q in enumerate(data["questions"]) if q.get("type") == "yesno" and q.get("exact_answer", "").lower() in ["yes", "no"]]
        return qs[:count] if count else qs
    else:
        df = pd.read_parquet("/mnt/d/Harsha/AoLM/project/data/pubmed_qa_pga_labeled.parquet")
        vm = {"yes": "A", "no": "B", "maybe": "C"}
        qs = [{"id": str(r.get("pubid", i)), "question": r["question"], "options": {"A": "Yes", "B": "No", "C": "Maybe"},
               "answer": vm.get(r["final_decision"].strip().lower(), "C")} 
              for i, r in df.iterrows()]
        return qs[:count] if count else qs

def answer_single(medrag_instance, llm, question, options, snippets):
    # Monkey patch MedRAG to use the EXACT Ollama model as ClinProof
    medrag_instance.generate = lambda messages, **kw: llm.generate(messages, temperature=0.0)
    
    ans, snippets_out, scores = medrag_instance.answer(question=question, options=options, snippets=snippets, k=25)
    
    # Parse output for answer_choice
    import re
    parsed_ans = "A"
    if isinstance(ans, str):
        match = re.search(r'\b([ABC])\b', ans)
        if match: parsed_ans = match.group(1).upper()
        raw_ans = ans
    else:
        raw_ans = str(ans)

    return parsed_ans, {"raw_output": raw_ans}, snippets

def run_eval(name, questions, llm, medrag_instance, moe, out_path, enable_pubmed=True, pubmed_mode="pmc"):
    print(f"\n{'='*70}\n  MedRAG | {name}  |  {len(questions)} questions\n{'='*70}\n")
    
    done = {}
    if CONFIG["resume"] and os.path.exists(out_path):
        with open(out_path, "r") as f:
            try: done = {r["id"]: r for r in json.load(f).get("results", [])}
            except: pass
        if done: print(f"Resuming from checkpoint ({len(done)} completed)")

    results = list(done.values())
    correct = sum(1 for r in results if r["correct"])

    for qi, q in enumerate(questions):
        if q["id"] in done: continue
        
        print(f"[{qi+1}/{len(questions)}] {q['question'][:50]}...", end=" ", flush=True)
        t = time.time()
        
        try:
            # 1. ClinProof Retrieval
            raw_docs, _ = moe.retrieve(q["question"], k1=CONFIG["k"], options=q["options"], enable_pubmed=enable_pubmed, pubmed_mode=pubmed_mode)
            
            # Format docs to match MedRAG snippet format: [{"id": idx, "title": d["title"], "content": d["content"]}]
            formatted_snippets = [{"id": f"doc_{j}", "title": d["title"], "content": d["content"]} for j, d in enumerate(raw_docs)]
            
            # 2. MedRAG Generation
            pred, res, snippets = answer_single(medrag_instance, llm, q["question"], q["options"], snippets=formatted_snippets)
            ctx = "\n".join([f"Docs [{j+1}] (Title: {c['title']}): {c['content']}" for j, c in enumerate(snippets)])
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            pred, res, ctx = "?", {}, "Error"

        ok = pred == q["answer"]
        if ok: correct += 1
        elapsed = time.time() - t
        print(f"{'✅' if ok else '❌'} pred={pred} gt={q['answer']} ({elapsed:.1f}s)")
        
        rec = {
            "id": q["id"],
            "question": q["question"],
            "gt_answer": q["answer"],
            "pred_answer": pred,
            "correct": ok,
            "time_seconds": elapsed,
            "retrieved_context": ctx,
            "reasoning_trace": res.get("raw_output", "")
        }
        results.append(rec)
        done[q["id"]] = rec
        
        if len(results) % 5 == 0:
            with open(out_path, "w") as f:
                json.dump({"config": CONFIG, "accuracy": correct/len(results), "results": results}, f, indent=2)

    acc = correct / len(questions) if questions else 0
    with open(out_path, "w") as f:
        json.dump({"config": CONFIG, "accuracy": acc, "results": results}, f, indent=2)
        
    print(f"\n  {name} Final Accuracy: {correct}/{len(questions)} ({acc*100:.1f}%)")
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["all", "bioasq", "pubmedqa"], default="all")
    args = parser.parse_args()

    os.makedirs("/mnt/d/Harsha/AoLM/project/clinproof/results", exist_ok=True)
    cfg = yaml.safe_load(open("/mnt/d/Harsha/AoLM/project/clinproof/config/default.yaml"))

    print("\nStarting MedRAG Evaluation with ClinProof Data Sources")
    print(f"Dataset selected: {args.dataset.upper()}")
    
    ll_model = OllamaLLM({"model": {"name": CONFIG["model"]}})

    # Setup ClinProof Retrievers exactly like eval_universal.py
    graph = GraphRetriever(cfg["kg"]["graph_path"], cfg, llm=ll_model) if CONFIG["use_graph"] else None
    moe   = MoERetriever(graph, None, None, cfg, ollama_client=ll_model.client)
    
    if CONFIG["use_pubmed"]:
        print("Loading PubMed MedCPT retriever...")
        moe.pubmed = PubMedDenseRetriever(cfg)

    import logging
    logging.getLogger('httpx').setLevel(logging.WARNING)
    os.environ["OPENAI_API_KEY"] = "dummy"

    # We use a dummy OpenAI name because initializing MedRAG with local HuggingFace weights installs gigabytes
    # By passing rag=False, we bypass the 30GB corpus download
    medrag = MedRAG(
        llm_name="OpenAI/gpt-3.5-turbo-16k", 
        rag=False 
    )
    # Manually turn RAG features back on to accept injected snippets
    medrag.rag = True
    medrag.retrieval_system = None

    b_path = f"/mnt/d/Harsha/AoLM/project/clinproof/results/{CONFIG['tag']}_bioasq.json"
    p_path = f"/mnt/d/Harsha/AoLM/project/clinproof/results/{CONFIG['tag']}_pubmedqa.json"
    
    b_acc, p_acc = None, None
    summary = "\n== Summary =="

    if args.dataset in ["all", "bioasq"]:
        b_acc = run_eval("BioASQ-Y/N", load_dataset("BioASQ", CONFIG["count"]), ll_model, medrag, moe, b_path, enable_pubmed=True, pubmed_mode="pubmed")
        summary += f"\nBioASQ: {b_acc:.1%}"
        
    if args.dataset in ["all", "pubmedqa"]:
        p_acc = run_eval("PubMedQA", load_dataset("PubMedQA", CONFIG["count"]), ll_model, medrag, moe, p_path, enable_pubmed=True, pubmed_mode="pmc")
        summary += f"\nPubMedQA: {p_acc:.1%}"

    print(summary)
