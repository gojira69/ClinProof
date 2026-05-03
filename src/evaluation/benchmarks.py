"""
ClinProof Benchmark Dataset Loaders
MedQA-US (local), MedMCQA, PubMedQA, MMLU-Medical (via HuggingFace datasets)
"""
import os, json, logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import project_path

log = logging.getLogger("benchmarks")
CACHE_DIR = Path(project_path("data", "benchmarks"))


def _first_existing(*candidates: Path) -> Path | None:
    return next((candidate for candidate in candidates if candidate.exists()), None)


def load_medqa_us(split="test", **kw):
    """Load from locally-downloaded MedQA dataset."""
    base = _first_existing(
        Path(project_path("data", "medqa-dataset", "data_clean", "questions", "US")),
        Path(project_path("data", "processed", "medqa-dataset", "data_clean", "questions", "US")),
    )
    if base is None:
        log.warning("MedQA-US dataset directory not found under data/")
        return []
    # Try common filename patterns
    candidates = [base/f"{split}.jsonl", base/f"US_{split}.jsonl"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        # Try any matching file
        matches = list(base.glob(f"*{split}*.jsonl")) if base.exists() else []
        path = matches[0] if matches else None
    if path is None:
        log.warning(f"MedQA-US {split} not found in {base}")
        return []
    data = []
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            options = item.get("options", {})
            answer = item.get("answer_idx", item.get("answer", ""))
            data.append({"question": item["question"], "options": options, "answer": answer, "meta": {"source": "medqa_us"}})
    log.info(f"MedQA-US {split}: {len(data)} questions")
    return data


def _hf_load(dataset_id, subset, split, cache_name, transform_fn, max_samples=None):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / cache_name
    if not cache.exists():
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_id, subset, split=split) if subset else load_dataset(dataset_id, split=split)
            with open(cache, "w") as f:
                for item in ds:
                    f.write(json.dumps(dict(item)) + "\n")
        except Exception as e:
            log.error(f"Download failed {dataset_id}: {e}"); return []
    data = []
    with open(cache) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples: break
            try: data.append(transform_fn(json.loads(line.strip())))
            except Exception: pass
    log.info(f"{cache_name}: {len(data)}")
    return data


def load_medmcqa(split="validation", max_samples=4183, **kw):
    opt_map = {0:"A",1:"B",2:"C",3:"D"}
    def tr(item):
        return {"question": item["question"], "options": {"A":item.get("opa",""),"B":item.get("opb",""),"C":item.get("opc",""),"D":item.get("opd","")}, "answer": opt_map.get(item.get("cop",0),"A"), "meta":{"source":"medmcqa"}}
    return _hf_load("openlifescienceai/medmcqa", None, split, f"medmcqa_{split}.jsonl", tr, max_samples)


def load_pubmedqa(split="train", max_samples=500, **kw):
    def tr(item):
        ans_map = {"yes":"A","no":"B","maybe":"C"}
        return {"question": item["question"], "options":{"A":"yes","B":"no","C":"maybe"}, "answer": ans_map.get(str(item.get("final_decision","maybe")).lower(),"C"), "meta":{"source":"pubmedqa"}}
    return _hf_load("qiaojin/PubMedQA", "pqa_labeled", "train", "pubmedqa.jsonl", tr, max_samples)


def load_mmlu_medical(split="test", **kw):
    subjects = ["anatomy","clinical_knowledge","college_biology","college_medicine","medical_genetics","professional_medicine"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"mmlu_med_{split}.jsonl"
    if not cache.exists():
        try:
            from datasets import load_dataset
            with open(cache, "w") as f:
                for sub in subjects:
                    try:
                        ds = load_dataset("cais/mmlu", sub, split=split)
                        for item in ds:
                            item["_sub"] = sub; f.write(json.dumps(dict(item))+"\n")
                    except Exception as e:
                        log.warning(f"MMLU {sub}: {e}")
        except Exception as e:
            log.error(f"MMLU download failed: {e}"); return []
    data = []
    with open(cache) as f:
        for line in f:
            item = json.loads(line.strip())
            choices = item.get("choices", [])
            options = {chr(65+i): c for i, c in enumerate(choices)}
            ai = item.get("answer", 0)
            answer = chr(65+ai) if isinstance(ai, int) else str(ai)
            data.append({"question": item["question"], "options": options, "answer": answer, "meta": {"source":"mmlu_med","subject":item.get("_sub","")}})
    log.info(f"MMLU Medical {split}: {len(data)}")
    return data


def load_bioasq(split="train", max_samples=500, **kw):
    def tr(item):
        qtype = item.get("type","yesno")
        if qtype == "yesno":
            opts = {"A":"yes","B":"no"}
            ans = "A" if item.get("exact_answer","yes") == "yes" else "B"
        else:
            opts = None; ideal = item.get("ideal_answer",[""]); ans = (ideal[0] if isinstance(ideal,list) else ideal)
        return {"question": item.get("body",""), "options": opts, "answer": ans, "meta": {"source":"bioasq","type":qtype}}
    return _hf_load("kroshan/BioASQ", None, "train", "bioasq.jsonl", tr, max_samples)


DATASET_LOADERS = {"medqa_us": load_medqa_us, "medmcqa": load_medmcqa, "pubmedqa": load_pubmedqa, "mmlu_med": load_mmlu_medical, "bioasq": load_bioasq}


def load_dataset_by_name(name, split="test", **kwargs):
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Options: {list(DATASET_LOADERS.keys())}")
    return DATASET_LOADERS[name](split=split, **kwargs)
