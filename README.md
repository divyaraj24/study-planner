# 📚 Studbud: Personalized Study Planner

**Studbud** is an AI-powered web application built using **Streamlit**, which helps students create **customized study plans** based on their individual needs and goals. The app utilizes a **fine-tuned BERT model** for text classification and **Google Gemini** for intelligent information extraction and plan generation.

---

## 🚀 Features

### 1. **Intelligent Goal Detection**
- Users input a natural-language study intent (e.g., "I want to study DSA for my upcoming exam").
- The system classifies the goal as **Exam**, **Project**, or **Mastery** using a BERT model.

### 2. **Information Extraction with Gemini**
- Extracts structured information from the user input such as:
  - Goal
  - Subject
  - Available study time
  - Current academic level
  - Learning preferences
  - Difficulty level

### 3. **Editable Study Form**
- Pre-filled form fields allow users to review and modify their information.
- Detects and warns about missing or unclear data.

### 4. **Dynamic Study Plan Generator**
- Generates a tailored plan based on:
  - Study scenario (Exam, Project, Mastery)
  - Time availability and difficulty
  - Preferred learning styles
- Includes:
  - Timeline breakdown
  - Daily workload
  - Strategy recommendations
---

## 💡 Technologies Used
- **Python**
- **Streamlit** (UI)
- **HuggingFace Transformers** (BERT-based classification)
- **Gemini 2.5 flash** (Information extraction + plan enhancement)
- **Dotenv** (Environment management)

---

## 🌟 Use Case
This app is ideal for:
- Students preparing for upcoming **exams**
- Learners managing **long-term mastery**
- Anyone organizing a **project-based learning** journey

---

### ⚠️ BERT Model Disclaimer

This project requires a fine-tuned BERT model directory named `bert_study_model` which is **not included** in the GitHub repository due to file size limitations (GitHub restricts files larger than 100MB).

### 📥 Download Instructions

Please manually download the model from the link below:

🔗 [Download `bert_study_model.zip`](https://drive.google.com/file/d/1id0M2myASpL34piavcAoKJamqN9_d6B9/view?usp=sharing)  

### 📦 Setup Instructions

1. Download the ZIP file from the link above.
2. Extract it into the root of your project directory so the structure looks like what is given below.

## 📁 Project Structure:

```
study-planner/
├── bert_study_model/               # Fine-tuned BERT model files
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
├── data/                           # All CSV or raw input/output data
│   ├── test_3way.csv
|   ├── bert_finetune_train.csv
│   └── test_results_3way.csv
├── StudyPlanner/                   # Source code package
│   ├── studbud.py                  # Streamlit app
│   └── test_model.py               # Model evaluation script
├── scripts/                        # For training, preprocessing, etc.
│   └── bert_finetune.ipynb         # Your Colab notebook or Python script
├── README.md                       # Project overview and usage
├── .gitignore                      # Ignored files
├── .env                            # Add environment variables like API keys
└── requirements.txt                # Dependencies for setup
```

# 🧠 Fine-Tuned BERT Model for Study Task Classification

## 🔍 Overview

This project involves fine-tuning a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** model for a **3-way text classification task** tailored to educational content. The goal of the model is to classify text-based user inputs into one of the following categories:

- **Exam** 📝 — Texts related to exam preparation or exam-related study.
- **Project** 💻 — Content indicating project-based learning or assignments.
- **Mastery** 📚 — Statements reflecting a desire for deep understanding or skill acquisition.

### ✅ Performance Highlights

After fine-tuning the BERT model on a balanced dataset of labeled study planning texts, the model achieved:

- **Overall Accuracy:** `95.51%`
- **Macro F1-Score:** `0.96`

### 📊 Classification Report

| Label    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| **Exam**     | 0.93      | 0.93   | 0.93     | 30      |
| **Mastery**  | 0.93      | 0.93   | 0.93     | 30      |
| **Project**  | 1.00      | 1.00   | 1.00     | 29      |
| **Overall**  |           |        | **0.96** | 89      |

### ⚙️ Model Architecture

- **Base Model:** `bert-base-uncased` from Hugging Face Transformers
- **Task Type:** Multi-class sequence classification (3 classes)
- **Token Limit:** Inputs truncated or padded to 128 tokens
- **Epochs:** Trained for 8 epochs
- **Training Batch Size:** 16
- **Evaluation Batch Size:** 8
- **Learning Rate:** `2e-5` with weight decay of `0.01`

### 🧪 Dataset

- Total Examples: `~89` (balanced across the 3 classes)
- Data Split: `80% training`, `20% validation`
- Labels encoded as:
  - `0` → Exam  
  - `1` → Project  
  - `2` → Mastery  

### 💾 Model Output

- Trained model and tokenizer saved to `bert_study_model/`
- Exported as a downloadable ZIP file for deployment or reuse

### 🎯 Next Goal

- As the BERT model was trained only on one line inputs the next step is to make a fine tuned model trained on a dataset with multiple line values.

---

## ⚠️ Disclaimer on Original Guided Project

The original guided project (`StudBud: Study Planner using BERT`) does not generate study plans using BERT. Instead, it passes a prompt to the base BERT model without fine-tuning, which simply picks a number (a class label). The study plan is then hardcoded based on that number — not generated by the model itself.

This project uses a **fine-tuned BERT model** to take raw user input and classify it into the correct scenario instead.

After which some information is extracted from raw user input in another form to generate a study plan.