from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "./bert_study_model"

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


# Example Classification
predicted_label = classify_text("I want to perfect my understanding of a subject")
print(f"Predicted label: {predicted_label[0]}, Confidence: {predicted_label[1]:.4f}")

# To add streamlit GUI...