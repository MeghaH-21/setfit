from setfit import SetFitModel
import json
import numpy as np

# Load trained SetFit model
model = SetFitModel.from_pretrained("setfit_model/")

# Load label mapping
with open("label_mapping.json", "r") as f:
    label_mapping = json.load(f)

# Convert mapping keys to integers if needed
# Ensure all IDs are integers for prediction lookup
id2label = {int(v): k for k, v in label_mapping.items()}

print("=== VoizPanda Intent Classification ===")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter your message: ")
    if user_input.lower() == "exit":
        print("Exiting...")
        break

    # Get predicted label probabilities
    probs = model.predict_proba([user_input])[0]  # Returns list of probabilities
    probs_np = np.array(probs)

    # Get top prediction index and confidence
    top_idx = int(probs_np.argmax())
    top_confidence = float(probs_np[top_idx])
    intent = id2label.get(top_idx, "Unknown")

    print(f"\nPredicted intent: {intent}")
    print(f"Confidence score: {top_confidence:.2f}\n")

    print("All label probabilities:")
    for idx, prob in enumerate(probs_np):
        # Optional: Only show probabilities above 0.01 for readability
        if prob >= 0.01:
            label = id2label.get(idx, f"Label_{idx}")
            print(f"{label}: {prob:.2f}")
    print("\n" + "-"*50 + "\n")
