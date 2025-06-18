import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

model_path = "./bert_study_model"

# Load test data
df = pd.read_csv("data/test_3way.csv")  # Ensure columns: 'text' and 'label'

# Load the fine-tuned BERT model and tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


# Prediction while ignoring Unidentified for now...

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    print(f"Predictions: {probs}")
    print(f"Logits: {logits}")

    # Get class indices excluding 'Unidentified' (assumed at index 3)
    filtered_probs = probs[0][:3]  # [Exam, Project, Mastery]
    predicted_idx = torch.argmax(filtered_probs).item()
    confidence = filtered_probs[predicted_idx].item()
    
    class_labels = ["Exam", "Project", "Mastery"]
    return class_labels[predicted_idx], confidence

# Run predictions
predictions = []
confidences = []

for _, row in df.iterrows():
    pred, conf = classify_text(row["text"])
    predictions.append(pred)
    confidences.append(conf)

df["predicted"] = predictions
df["confidence"] = confidences

# Evaluation
y_true = df["label"]
y_pred = df["predicted"]

accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, digits=2)

print(f"✅ Accuracy: {accuracy * 100:.2f}%")
print("📊 Classification Report:\n")
print(report)

# Optionally save the result
df.to_csv("data/test_results_3way.csv", index=False)

