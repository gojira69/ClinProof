"""
ClinProof Evaluation Metrics: accuracy + citation evaluation
"""
import os, re, json, logging
import numpy as np

log = logging.getLogger("metrics")


def normalize_answer(ans):
    if not ans: return ""
    ans = str(ans).strip().upper()
    m = re.search(r'\b([ABCDE])\b', ans)
    if m: return m.group(1)
    if "YES" in ans: return "A"
    if "NO" in ans: return "B"
    return ans[0] if ans else ""


def compute_accuracy(results):
    correct, total, skipped = 0, 0, 0
    for r in results:
        gt, pred = r.get("ground_truth",""), r.get("answer_choice","")
        if not gt or not pred: skipped += 1; continue
        total += 1
        if normalize_answer(str(gt)) == normalize_answer(str(pred)): correct += 1
    return {"accuracy": correct/total if total else 0.0, "correct": correct, "total": total, "skipped": skipped}


class CitationEvaluator:
    """Gemini or local NLI for citation recall/precision."""
    def __init__(self, config):
        self.use_gemini = config.get("evaluation",{}).get("judge","gemini") == "gemini"
        self.gemini_model = config.get("evaluation",{}).get("gemini_model","gemini-2.0-flash")
        self._client = None; self._nli = None
        if self.use_gemini: self._init_gemini(config)
        else: self._init_nli()

    def _init_gemini(self, config):
        key = os.environ.get("GOOGLE_API_KEY") or config.get("evaluation",{}).get("gemini_api_key","")
        if not key: log.warning("No GOOGLE_API_KEY - using local NLI"); self.use_gemini=False; self._init_nli(); return
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            self._client = genai.GenerativeModel(self.gemini_model, generation_config={"temperature":0,"max_output_tokens":32})
            log.info(f"Gemini judge: {self.gemini_model}")
        except Exception as e:
            log.warning(f"Gemini init failed: {e}"); self.use_gemini=False; self._init_nli()

    def _init_nli(self):
        try:
            from sentence_transformers import CrossEncoder
            self._nli = CrossEncoder("cross-encoder/nli-deberta-v3-small", max_length=512)
            log.info("Local NLI initialized")
        except Exception as e:
            log.warning(f"Local NLI init failed: {e}")

    def supports(self, doc_text, statement):
        if self.use_gemini and self._client:
            try:
                resp = self._client.generate_content(f"Document: {doc_text[:300]}\nStatement: {statement}\nDoes the document support this statement? yes or no.")
                return 1.0 if "yes" in resp.text.strip().lower() else 0.0
            except Exception: return 0.5
        elif self._nli:
            try:
                scores = self._nli.predict([(doc_text[:512], statement)])
                probs = scores[0] if hasattr(scores[0],"__iter__") else [0,scores[0],0]
                return float(probs[1] > 0.5)
            except Exception: return 0.5
        return 0.5

    def evaluate_citations(self, result):
        answer = result.get("answer",""); cited = result.get("cited_docs",{})
        if not answer or not cited: return {"recall":0.0,"precision":0.0,"n":0}
        pattern = re.compile(r'\[(\d+)\]')
        sents = re.split(r'(?<=[.!?])\s+', answer)
        recalls, precisions = [], []
        for s in sents:
            cites = list(set(pattern.findall(s)))
            if not cites: continue
            stmt = pattern.sub("",s).strip()
            if len(stmt) < 10: continue
            valid = [c for c in cites if c in cited]
            if not valid: continue
            combined = " ".join(cited[c].get("content","")[:300] for c in valid)
            recalls.append(self.supports(combined, stmt))
            for c in valid:
                precisions.append(self.supports(cited[c].get("content",""), stmt))
        return {"recall": float(np.mean(recalls)) if recalls else 0.0, "precision": float(np.mean(precisions)) if precisions else 0.0, "n": len(recalls)}


def evaluate_results(results, config, evaluate_citations=True):
    acc = compute_accuracy(results)
    out = {"accuracy": acc}
    if not evaluate_citations:
        return out
    ev = CitationEvaluator(config)
    from tqdm import tqdm
    recalls, precisions = [], []
    for r in tqdm(results, desc="Evaluating citations"):
        if r.get("cited_docs"):
            ce = ev.evaluate_citations(r)
            recalls.append(ce["recall"]); precisions.append(ce["precision"])
    out["citation"] = {"recall": float(np.mean(recalls)) if recalls else 0.0, "precision": float(np.mean(precisions)) if precisions else 0.0, "n_evaluated": len(recalls)}
    return out


def print_results_table(metrics, system_name="ClinProof"):
    print(f"\n{'='*55}\n  {system_name}\n{'='*55}")
    a = metrics.get("accuracy",{})
    print(f"  Accuracy: {a.get('accuracy',0):.4f}  ({a.get('correct',0)}/{a.get('total',0)})")
    c = metrics.get("citation",{})
    if c:
        print(f"  Citation Recall:    {c.get('recall',0):.4f}")
        print(f"  Citation Precision: {c.get('precision',0):.4f}")
    print(f"{'='*55}\n")
