"""
ClinProof Evaluation Metrics: classification + citation evaluation
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


def compute_classification_metrics(results, pred_field="pred_answer", gt_field="gt_answer"):
    """Compute accuracy plus macro precision/recall/F1 from result records."""
    pairs = []
    for r in results:
        gt = normalize_answer(r.get(gt_field, ""))
        pred = normalize_answer(r.get(pred_field, ""))
        if gt and pred:
            pairs.append((gt, pred))

    labels = sorted({gt for gt, _ in pairs} | {pred for _, pred in pairs})
    per_class = {}

    for label in labels:
        tp = sum(1 for gt, pred in pairs if gt == label and pred == label)
        fp = sum(1 for gt, pred in pairs if gt != label and pred == label)
        fn = sum(1 for gt, pred in pairs if gt == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "tp": tp,
        }

    total = len(pairs)
    correct = sum(1 for gt, pred in pairs if gt == pred)
    macro_precision = (float(np.mean([m["precision"] for m in per_class.values()]))
                       if per_class else 0.0)
    macro_recall = (float(np.mean([m["recall"] for m in per_class.values()]))
                    if per_class else 0.0)
    macro_f1 = (float(np.mean([m["f1"] for m in per_class.values()]))
                if per_class else 0.0)

    return {
        "accuracy": correct / total if total else 0.0,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
        "correct": correct,
        "total": total,
        "per_class": per_class,
    }


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
    cls = metrics.get("classification", {})
    if cls:
        print(f"  Macro Precision: {cls.get('precision',0):.4f}")
        print(f"  Macro Recall:    {cls.get('recall',0):.4f}")
        print(f"  Macro F1:        {cls.get('f1',0):.4f}")
    c = metrics.get("citation",{})
    if c:
        print(f"  Citation Recall:    {c.get('recall',0):.4f}")
        print(f"  Citation Precision: {c.get('precision',0):.4f}")
    print(f"{'='*55}\n")
