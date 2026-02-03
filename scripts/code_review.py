from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import os

# Load pretrained CodeBERT (random classifier head unless fine-tuned)
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained(
    "microsoft/codebert-base", num_labels=2
)

def review_code(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()
    inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=-1).item()
    return prediction

# Scan all Python files in repo
issues_found = False
for root, dirs, files in os.walk("."):
    for file in files:
        if file.endswith(".py"):
            result = review_code(os.path.join(root, file))
            if result == 1:  # Issue found
                issues_found = True
                print(f"⚠️ Potential issue in {file}")

if issues_found:
    exit(1)  # Fail pipeline
else:
    print("✅ Code looks clean!")
