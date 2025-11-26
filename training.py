import json
from setfit import SetFitTrainer, SetFitModel
from datasets import Dataset

# ===========================
# Step 1: Load your dataset
# ===========================
with open("education_industry.json", "r") as f:  # your dataset file
    data = json.load(f)

# ===========================
# Step 2: Prepare training data
# ===========================
texts = []
labels = []

for persona in data["personas"]:
    for use_case, examples in persona["use_cases"].items():
        for ex in examples:
            texts.append(ex)
            labels.append(use_case)

print(f"Total training examples: {len(texts)}")

# Create HuggingFace Dataset
train_dataset = Dataset.from_dict({"text": texts, "label": labels})

# ===========================
# Step 3: Initialize SetFit model
# ===========================
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

# ===========================
# Step 4: Initialize trainer
# ===========================
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=None,  # optional: you can split some for validation
    metric="accuracy",
)

# ===========================
# Step 5: Train the model
# ===========================
trainer.train()

# ===========================
# Step 6: Save the trained model and label mapping
# ===========================
#trainer.model.save_pretrained("setfit_model/")
trainer.model.save_pretrained("setfit_model/")
label_mapping = {label: idx for idx, label in enumerate(sorted(set(labels)))}
with open("label_mapping.json", "w") as f:
    json.dump(label_mapping, f, indent=2)

print("Training complete! Model saved in 'setfit_model/' and labels in 'label_mapping.json'.")
