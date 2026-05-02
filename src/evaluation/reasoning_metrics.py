"""
ClinProof Reasoning & Justification Metrics
============================================

All metrics are sourced from the literature reviewed for this project:

  ROUGE-L          — MedRAG (§6.1), MedCite (§3.3)
  BERTScore F1     — MedRAG (§6.1)
  MAUVE            — MedCite (§3.3, Pillutla et al. 2021)
  Faithfulness     — MedCite §4.3 (NLI: does reasoning → conclusion)
  Evidence Grounding — MedCite §4.3 (NLI: does context → reasoning)
  Citation Recall  — MedCite §3.3 (Recall: all facts in reasoning backed by docs)
  Citation Precision — MedCite §3.3 (Precision: each cited doc backs claim)
  Macro-F1         — BioASQ, SciFact standard evaluation protocol
  Accuracy (EM)    — Universal (MedCite §3.3, MedRAG §6.1)

Usage:
    from src.evaluation.reasoning_metrics import ReasoningMetrics
    rm = ReasoningMetrics()
    scores = rm.score_results(results_list)
"""
import re
import logging
import json
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

log = logging.getLogger("reasoning_metrics")


# ── Optional heavy imports (loaded lazily) ────────────────────────────────────

def _rouge():
    try:
        from rouge_score import rouge_scorer
        return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        log.warning("rouge_score not installed. pip install rouge-score")
        return None


def _bert_score():
    try:
        import bert_score as bs
        return bs
    except ImportError:
        log.warning("bert_score not installed. pip install bert-score")
        return None


def _nli_model():
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/nli-deberta-v3-small", max_length=512)
        log.info("NLI CrossEncoder loaded.")
        return model
    except ImportError:
        log.warning("sentence_transformers not installed. pip install sentence-transformers")
        return None
    except Exception as e:
        log.warning(f"NLI model load failed: {e}")
        return None


def _mauve_fn():
    try:
        import mauve
        return mauve
    except ImportError:
        log.warning("mauve not installed. pip install mauve-text")
        return None


# ── Label normalisation ───────────────────────────────────────────────────────

def _normalize_label(label: str) -> str:
    """Normalise predicted/GT labels to a canonical short string."""
    s = str(label).strip().upper()
    mapping = {
        "SUPPORTED": "SUP", "SUPPORT": "SUP", "A": "SUP",
        "REFUTED": "REF", "CONTRADICT": "REF", "CONTRADICTED": "REF", "B": "REF",
        "NOT ENOUGH INFORMATION": "NEI", "NEI": "NEI", "C": "NEI",
        "YES": "YES",
        "NO": "NO",
        "TRUE": "TRUE",
        "FALSE": "FALSE",
        "MIXTURE": "MIX",
    }
    return mapping.get(s, s)


# ── Core class ────────────────────────────────────────────────────────────────

class ReasoningMetrics:
    """
    Compute all literature-sourced reasoning metrics from saved result records.

    Each result record is expected to have at minimum:
        - reasoning_traces: list of {step_by_step_thinking: str, choice: str, ...}
        - retrieved_context: str
        - correct: bool
        - gt_answer/gt_label: str
        - pred_answer/pred_label: str
        - vote_distribution: dict

    Reference answer (for ROUGE/BERTScore) is approximated as the reasoning
    trace from correct predictions, since no gold-standard reasoning exists.
    When no reference is available, these metrics are skipped.
    """

    def __init__(self, use_nli: bool = True, use_bertscore: bool = False,
                 use_mauve: bool = False):
        """
        Args:
            use_nli:       Compute Faithfulness + Evidence Grounding (slow, requires model).
            use_bertscore: Compute BERTScore (very slow, downloads microsoft/deberta-xlarge).
            use_mauve:     Compute MAUVE (requires mauve-text package).
        """
        self.use_nli = use_nli
        self.use_bertscore = use_bertscore
        self.use_mauve = use_mauve

        self._rouge = _rouge()
        self._nli   = _nli_model() if use_nli else None
        self._bs    = _bert_score() if use_bertscore else None
        self._mauve = _mauve_fn()  if use_mauve else None

    # ── ROUGE-L ───────────────────────────────────────────────────────────────

    def rouge_l(self, hypothesis: str, reference: str) -> float:
        """ROUGE-L F1 between generated reasoning and reference reasoning."""
        if not self._rouge or not hypothesis or not reference:
            return float("nan")
        try:
            scores = self._rouge.score(reference, hypothesis)
            return float(scores["rougeL"].fmeasure)
        except Exception as e:
            log.debug(f"ROUGE-L error: {e}")
            return float("nan")

    # ── BERTScore ─────────────────────────────────────────────────────────────

    def bertscore_f1(self, hypotheses: list[str], references: list[str],
                     lang: str = "en") -> list[float]:
        """
        BERTScore F1 (batch), as used in MedRAG §6.1.
        Returns list of floats, one per (hyp, ref) pair.
        """
        if not self._bs or not hypotheses:
            return [float("nan")] * len(hypotheses)
        try:
            P, R, F1 = self._bs.score(
                hypotheses, references, lang=lang,
                model_type="microsoft/deberta-xlarge-mnli",
                verbose=False, batch_size=8
            )
            return [float(f) for f in F1.tolist()]
        except Exception as e:
            log.warning(f"BERTScore error: {e}")
            return [float("nan")] * len(hypotheses)

    # ── NLI-based: Faithfulness ───────────────────────────────────────────────

    def faithfulness(self, reasoning: str, conclusion_label: str) -> float:
        """
        Faithfulness (MedCite §4.3): Does the reasoning entail the stated conclusion?

        NLI: premise = step_by_step_thinking
             hypothesis = "The answer is {label}"
        Returns entailment probability in [0,1].
        """
        if not self._nli or not reasoning or not conclusion_label:
            return float("nan")
        hypothesis = f"The answer to this medical question is: {conclusion_label}."
        try:
            scores = self._nli.predict(
                [(reasoning[:512], hypothesis)], apply_softmax=True
            )
            # DeBERTa NLI labels: contradiction=0, neutral=1, entailment=2
            probs = scores[0]
            return float(probs[2])   # entailment probability
        except Exception as e:
            log.debug(f"Faithfulness NLI error: {e}")
            return float("nan")

    # ── NLI-based: Evidence Grounding ─────────────────────────────────────────

    def evidence_grounding(self, context: str, reasoning: str) -> float:
        """
        Evidence Grounding (MedCite §4.3 Re-retrieval+NLI rerank):
        Does the retrieved context support the generated reasoning?

        NLI: premise = retrieved_context (truncated to 512 tokens)
             hypothesis = step_by_step_thinking (truncated)
        Returns entailment probability in [0,1].
        """
        if not self._nli or not context or not reasoning:
            return float("nan")
        try:
            scores = self._nli.predict(
                [(context[:512], reasoning[:256])], apply_softmax=True
            )
            probs = scores[0]
            return float(probs[2])   # entailment probability
        except Exception as e:
            log.debug(f"Evidence grounding NLI error: {e}")
            return float("nan")

    # ── Citation Recall & Precision ───────────────────────────────────────────

    def citation_recall_precision(self, reasoning: str, context_docs: list[str]
                                   ) -> dict:
        """
        Citation Recall & Precision (MedCite §3.3).

        Recall:    Is every factual claim in the reasoning supported by at least
                   one retrieved doc (concatenated)?
        Precision: For each retrieved doc, does it independently support the reasoning?

        Both use NLI (premise=doc, hypothesis=reasoning-sentence).
        Returns {recall, precision, f1, n_sentences}.
        """
        if not self._nli or not reasoning or not context_docs:
            return {"recall": float("nan"), "precision": float("nan"),
                    "f1": float("nan"), "n_sentences": 0}

        # Split reasoning into sentences
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", reasoning)
                     if len(s.strip()) > 20]
        if not sentences:
            return {"recall": float("nan"), "precision": float("nan"),
                    "f1": float("nan"), "n_sentences": 0}

        # Recall: concat all docs → supports sentence?
        combined_ctx = " ".join(d[:400] for d in context_docs[:5])
        recall_scores = []
        for sent in sentences[:8]:   # cap at 8 sentences for speed
            try:
                sc = self._nli.predict(
                    [(combined_ctx[:512], sent)], apply_softmax=True
                )
                # MedCite Attr: 1 if fully supported, 0.5 if partial, 0 otherwise
                ent_prob = float(sc[0][2])
                contra_prob = float(sc[0][0])
                if ent_prob > 0.6:
                    recall_scores.append(1.0)
                elif ent_prob > 0.3:
                    recall_scores.append(0.5)    # partial (as per MedCite §3.3)
                else:
                    recall_scores.append(0.0)
            except Exception:
                recall_scores.append(float("nan"))

        # Precision: each doc independently → supports reasoning?
        precision_scores = []
        reasoning_short = " ".join(sentences[:3])[:256]
        for doc_text in context_docs[:5]:
            try:
                sc = self._nli.predict(
                    [(doc_text[:512], reasoning_short)], apply_softmax=True
                )
                # Precision: 1 if doc supports even partially (MedCite §3.3)
                ent_prob = float(sc[0][2])
                precision_scores.append(1.0 if ent_prob > 0.3 else 0.0)
            except Exception:
                precision_scores.append(float("nan"))

        valid_recall = [s for s in recall_scores if not np.isnan(s)]
        valid_prec   = [s for s in precision_scores if not np.isnan(s)]
        R = float(np.mean(valid_recall)) if valid_recall else float("nan")
        P = float(np.mean(valid_prec))   if valid_prec   else float("nan")
        F1 = (2 * P * R / (P + R)) if (not np.isnan(P) and not np.isnan(R)
                                         and (P + R) > 0) else float("nan")
        return {"recall": R, "precision": P, "f1": F1,
                "n_sentences": len(sentences)}

    # ── Macro-F1 ──────────────────────────────────────────────────────────────

    @staticmethod
    def macro_f1(results: list[dict],
                 pred_field: str = "pred_label",
                 gt_field: str   = "gt_label") -> dict:
        """
        Macro-F1 and per-class P/R/F1 — standard for BioASQ/SciFact (imbalanced).

        Returns:
            {macro_f1, per_class: {label: {precision, recall, f1, support}}}
        """
        classes = sorted(set(str(r.get(gt_field, r.get("gt_answer", "")))
                             for r in results))
        per_class = {}
        for cls in classes:
            tp = sum(1 for r in results
                     if str(r.get(gt_field, r.get("gt_answer", ""))) == cls
                     and str(r.get(pred_field, r.get("pred_answer", ""))) == cls)
            fp = sum(1 for r in results
                     if str(r.get(pred_field, r.get("pred_answer", ""))) == cls
                     and str(r.get(gt_field, r.get("gt_answer", ""))) != cls)
            fn = sum(1 for r in results
                     if str(r.get(gt_field, r.get("gt_answer", ""))) == cls
                     and str(r.get(pred_field, r.get("pred_answer", ""))) != cls)
            P  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            R  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
            per_class[cls] = {"precision": P, "recall": R, "f1": F1,
                               "support": tp + fn}

        macro = float(np.mean([v["f1"] for v in per_class.values()])) if per_class else 0.0
        return {"macro_f1": macro, "per_class": per_class}

    # ── Vote Confidence ───────────────────────────────────────────────────────

    @staticmethod
    def vote_confidence(results: list[dict], n_votes: int) -> dict:
        """
        Analyse self-consistency voting confidence.
        unanimous_rate: fraction of questions with all votes agreeing.
        avg_winner_frac: mean fraction of votes the winning choice received.
        """
        unanimous, total, winner_fracs = 0, 0, []
        for r in results:
            vd = r.get("vote_distribution", {})
            if not vd:
                continue
            total += 1
            max_votes = max(vd.values())
            if max_votes == n_votes:
                unanimous += 1
            winner_fracs.append(max_votes / n_votes)
        return {
            "unanimous_rate": unanimous / total if total else 0.0,
            "avg_winner_frac": float(np.mean(winner_fracs)) if winner_fracs else 0.0,
            "n_evaluated": total,
        }

    # ── MAUVE ─────────────────────────────────────────────────────────────────

    def mauve_score(self, generated_texts: list[str],
                    reference_texts: list[str]) -> float:
        """
        MAUVE (MedCite §3.3, Pillutla et al. 2021).
        Measures distributional gap between generated reasoning and references.
        Requires mauve-text package and a GPU for reasonable speed.
        """
        if not self._mauve or not generated_texts or not reference_texts:
            return float("nan")
        try:
            out = self._mauve.compute_mauve(
                p_text=reference_texts,
                q_text=generated_texts,
                device_id=0,
                max_text_length=256,
                verbose=False,
            )
            return float(out.mauve)
        except Exception as e:
            log.warning(f"MAUVE error: {e}")
            return float("nan")

    # ── Full result scoring ───────────────────────────────────────────────────

    def score_results(self, results: list[dict],
                      n_votes: int = 3,
                      reference_results: Optional[list[dict]] = None,
                      max_nli_samples: int = 100) -> dict:
        """
        Compute all metrics for a list of result records.

        Args:
            results:          List of result dicts from eval_all.py.
            n_votes:          Number of self-consistency votes used.
            reference_results: If provided, use their reasoning as reference
                               for ROUGE-L/BERTScore. Otherwise, use correct
                               predictions as proxy references.
            max_nli_samples:  Cap on NLI computations (for speed).

        Returns:
            dict with all metric scores.
        """
        from tqdm import tqdm

        out = {}

        # ── 1. Macro-F1 & per-class ──────────────────────────────────────────
        out["macro_f1_report"] = self.macro_f1(results)
        log.info(f"Macro-F1: {out['macro_f1_report']['macro_f1']:.3f}")

        # ── 2. Vote confidence ───────────────────────────────────────────────
        out["vote_confidence"] = self.vote_confidence(results, n_votes)

        # ── 3. ROUGE-L on reasoning ──────────────────────────────────────────
        if self._rouge:
            # Reference: use reasoning from correct predictions as proxy
            correct_reasoning = [
                self._get_reasoning(r)
                for r in results if r.get("correct") and self._get_reasoning(r)
            ]
            if not correct_reasoning:
                log.warning("No correct predictions to use as ROUGE reference.")
                out["rouge_l"] = {"mean": float("nan"), "n": 0}
            else:
                ref_pool = correct_reasoning  # rotate reference
                scores = []
                for i, r in enumerate(results):
                    hyp = self._get_reasoning(r)
                    if not hyp:
                        continue
                    ref = ref_pool[i % len(ref_pool)]
                    scores.append(self.rouge_l(hyp, ref))
                valid = [s for s in scores if not np.isnan(s)]
                out["rouge_l"] = {
                    "mean": float(np.mean(valid)) if valid else float("nan"),
                    "n": len(valid),
                    "correct_mean": float(np.mean([
                        s for s, r in zip(scores, results)
                        if r.get("correct") and not np.isnan(s)
                    ])) if valid else float("nan"),
                    "wrong_mean": float(np.mean([
                        s for s, r in zip(scores, results)
                        if not r.get("correct") and not np.isnan(s)
                    ])) if valid else float("nan"),
                }

        # ── 4. NLI-based Faithfulness + Evidence Grounding ───────────────────
        if self._nli:
            sample = results[:max_nli_samples]
            faith_scores, eg_scores = [], []
            for r in tqdm(sample, desc="NLI metrics", leave=False):
                reasoning = self._get_reasoning(r)
                context   = r.get("retrieved_context", "")
                pred_label = r.get("pred_label", r.get("pred_answer", ""))
                if reasoning:
                    faith_scores.append(self.faithfulness(reasoning, pred_label))
                    if context:
                        eg_scores.append(self.evidence_grounding(context, reasoning))

            def _safe_mean(arr, mask=None):
                vals = arr if mask is None else [v for v, r in zip(arr, sample[:len(arr)]) if mask(r)]
                vals = [v for v in vals if not np.isnan(v)]
                return float(np.mean(vals)) if vals else float("nan")

            out["faithfulness"] = {
                "mean": _safe_mean(faith_scores),
                "correct_mean": _safe_mean(faith_scores,
                    lambda r: r.get("correct")),
                "wrong_mean": _safe_mean(faith_scores,
                    lambda r: not r.get("correct")),
                "n": len([s for s in faith_scores if not np.isnan(s)]),
            }
            out["evidence_grounding"] = {
                "mean": _safe_mean(eg_scores),
                "n": len([s for s in eg_scores if not np.isnan(s)]),
            }
            log.info(f"Faithfulness: {out['faithfulness']['mean']:.3f}, "
                     f"EvidGround: {out['evidence_grounding']['mean']:.3f}")

        # ── 5. Citation Recall/Precision ─────────────────────────────────────
        if self._nli:
            sample = results[:max_nli_samples]
            cite_recs, cite_precs, cite_f1s = [], [], []
            for r in tqdm(sample, desc="Citation metrics", leave=False):
                reasoning = self._get_reasoning(r)
                context   = r.get("retrieved_context", "")
                if not reasoning or not context:
                    continue
                # Split context into pseudo-docs by double newlines
                docs = [p.strip() for p in context.split("\n\n") if len(p.strip()) > 30]
                cr = self.citation_recall_precision(reasoning, docs)
                if not np.isnan(cr["recall"]):
                    cite_recs.append(cr["recall"])
                if not np.isnan(cr["precision"]):
                    cite_precs.append(cr["precision"])
                if not np.isnan(cr["f1"]):
                    cite_f1s.append(cr["f1"])
            out["citation"] = {
                "recall":    float(np.mean(cite_recs))  if cite_recs  else float("nan"),
                "precision": float(np.mean(cite_precs)) if cite_precs else float("nan"),
                "f1":        float(np.mean(cite_f1s))   if cite_f1s   else float("nan"),
                "n":         len(cite_recs),
            }

        return out

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_reasoning(result: dict) -> str:
        """Extract best reasoning trace from a result record."""
        # reasoning_traces is a list of {step_by_step_thinking, choice, ...}
        traces = result.get("reasoning_traces", [])
        if traces:
            # Use the trace that matches the final answer
            final = result.get("pred_answer", "")
            for t in traces:
                if t.get("choice") == final:
                    text = t.get("step_by_step_thinking", "")
                    if len(text) > 30:
                        return text
            # Fallback: first trace
            return traces[0].get("step_by_step_thinking", "")
        return ""


# ── Standalone CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json as _json, sys

    parser = argparse.ArgumentParser(
        description="Compute reasoning metrics on a saved eval_all.py result JSON."
    )
    parser.add_argument("result_file", help="Path to result JSON file")
    parser.add_argument("--no-nli",  action="store_true", help="Skip NLI metrics")
    parser.add_argument("--bertscore", action="store_true", help="Enable BERTScore (slow)")
    parser.add_argument("--mauve",     action="store_true", help="Enable MAUVE")
    parser.add_argument("--max-nli",   type=int, default=100)
    args = parser.parse_args()

    with open(args.result_file, encoding="utf-8") as f:
        data = _json.load(f)
    results  = data.get("results", [])
    n_votes  = data.get("config", {}).get("votes", 3)
    print(f"Loaded {len(results)} results, n_votes={n_votes}")

    rm = ReasoningMetrics(
        use_nli=not args.no_nli,
        use_bertscore=args.bertscore,
        use_mauve=args.mauve,
    )
    scores = rm.score_results(results, n_votes=n_votes,
                              max_nli_samples=args.max_nli)

    print("\n=== REASONING METRICS ===")
    print(_json.dumps(scores, indent=2, default=str))
