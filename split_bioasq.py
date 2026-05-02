import json
import random

random.seed(42)

input_path = '/mnt/d/Harsha/AoLM/ClinProof/data/processed/BioASQ-training7b/trainining7b.json'
out_train = '/mnt/d/Harsha/AoLM/ClinProof/data/processed/BioASQ-training7b/train.json'
out_test = '/mnt/d/Harsha/AoLM/ClinProof/data/processed/BioASQ-training7b/test.json'

print("Loading BioASQ data...")
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

questions = data.get('questions', [])
print(f'Total questions loaded: {len(questions)}')

# Filter for yes/no questions to be sure if we want, but letting them all split is fine
random.shuffle(questions)
split_idx = int(len(questions) * 0.8)
train_qs = questions[:split_idx]
test_qs = questions[split_idx:]

data_train = {'questions': train_qs}
data_test = {'questions': test_qs}

print("Saving train split...")
with open(out_train, 'w', encoding='utf-8') as f:
    json.dump(data_train, f, indent=2)

print("Saving test split...")
with open(out_test, 'w', encoding='utf-8') as f:
    json.dump(data_test, f, indent=2)

print(f'Split done! Train: {len(train_qs)} | Test: {len(test_qs)}')
