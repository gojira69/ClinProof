import json
import random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import project_path

random.seed(42)

input_path = project_path("data", "BioASQ-training13b", "training13b.json")
out_train = project_path("data", "BioASQ-training13b", "train.json")
out_test = project_path("data", "BioASQ-training13b", "test.json")

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
