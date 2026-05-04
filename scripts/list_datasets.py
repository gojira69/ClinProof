"""Count sizes of all available datasets."""
import json, os, glob
import pandas as pd

base = "/mnt/d/Harsha/AoLM/ClinProof/data/processed"

print("\n=== Available Datasets & Sizes ===\n")

# HealthFC
try:
    hfc_test  = pd.read_csv(f"{base}/healthfc_test.csv")
    hfc_all   = pd.read_csv(f"{base}/healthfc.csv")
    print(f"HealthFC      test={len(hfc_test)}  full={len(hfc_all)}")
    print(f"  Labels: {dict(hfc_test['label'].value_counts())}")
except Exception as e:
    print(f"HealthFC error: {e}")

# SciFact
try:
    sf_test  = pd.read_csv(f"{base}/scifact/claims_test.csv")
    sf_train = pd.read_csv(f"{base}/scifact/claims_train.csv")
    sf_val   = pd.read_csv(f"{base}/scifact/claims_validation.csv")
    print(f"\nSciFact       test={len(sf_test)}  train={len(sf_train)}  val={len(sf_val)}")
    if 'label' in sf_test.columns:
        print(f"  Labels: {dict(sf_test['label'].value_counts())}")
except Exception as e:
    print(f"\nSciFact error: {e}")

# MedQA
try:
    medqa_dir = f"{base}/medqa-dataset/data_clean"
    files = os.listdir(medqa_dir)
    print(f"\nMedQA         files: {files}")
    for fname in files:
        path = os.path.join(medqa_dir, fname)
        if fname.endswith(".jsonl"):
            lines = open(path).readlines()
            print(f"  {fname}: {len(lines)} questions")
        elif fname.endswith(".json"):
            d = json.load(open(path))
            print(f"  {fname}: {len(d)} entries")
except Exception as e:
    print(f"\nMedQA error: {e}")

# MedChangeQA / MedRevQA
try:
    medrev = pd.read_csv(f"{base}/MedChange-main/MedRevQA.csv")
    print(f"\nMedRevQA      total={len(medrev)}")
    print(f"  Cols: {list(medrev.columns[:6])}")
    if 'label' in medrev.columns:
        print(f"  Labels: {dict(medrev['label'].value_counts())}")
except Exception as e:
    print(f"\nMedRevQA error: {e}")

try:
    datasets_dir = f"{base}/MedChange-main/Datasets"
    for fname in os.listdir(datasets_dir):
        path = os.path.join(datasets_dir, fname)
        if fname.endswith(".csv"):
            df = pd.read_csv(path)
            print(f"  MedChange/{fname}: {len(df)} rows")
        elif fname.endswith(".jsonl") or fname.endswith(".json"):
            lines = open(path).readlines()
            print(f"  MedChange/{fname}: {len(lines)} lines")
except Exception as e:
    print(f"MedChange datasets error: {e}")

# BioASQ
try:
    with open(f"{base}/BioASQ-training13b/training13b.json") as f:
        bioasq = json.load(f)
    qs = bioasq.get("questions", [])
    yn_qs = [q for q in qs if q.get("type") == "yesno"]
    print(f"\nBioASQ-13b    total={len(qs)}  yes/no={len(yn_qs)}")
except Exception as e:
    print(f"\nBioASQ error: {e}")

# PubMedQA
try:
    pqa = pd.read_parquet(f"{base}/pubmed_qa_pga_labeled.parquet")
    print(f"\nPubMedQA      total={len(pqa)}")
    if 'final_decision' in pqa.columns:
        print(f"  Labels: {dict(pqa['final_decision'].value_counts())}")
except Exception as e:
    print(f"\nPubMedQA error: {e}")

print("\n=== Datasets in eval_all.py ===")
print("  bioasq, healthfc, scifact, medchangeqa, medqa, pubmedqa")
