import os
from google import genai
from dotenv import load_dotenv
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

load_dotenv()

# Ensure the correct Python environment is being used (for streamlit as well)
# Uncomment the following lines to debug the environment
# st.write("Python executable:", sys.executable)
# st.write("sys.path:", sys.path)
# st.write("Transformers path:", transformers.__file__)
# st.write("Transformers version:", transformers.__version__)

model_path = "./bert_study_model"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

@st.cache_resource
def load_model():
    model_path = "./bert_study_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map=None)
    return tokenizer, model


tokenizer, model = load_model()

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    #print(f"Predictions: {probs}")
    #print(f"Logits: {logits}")

    # Get class indices excluding 'Unidentified' (assumed at index 3)
    filtered_probs = probs[0][:3]  # [Exam, Project, Mastery]
    predicted_idx = torch.argmax(filtered_probs).item()
    confidence = filtered_probs[predicted_idx].item()
    
    class_labels = ["Exam", "Project", "Mastery"]
    return class_labels[predicted_idx], confidence

def extract_study_info(text):
    prompt = f"""
Extract the following details from the text and return as stated below:
- Goal i.e. to prepare for an exam, complete a project, or achieve mastery
- Subject
- Study time available in days
- Study time available in hours
- Current Grade/Year
- Learning Style (visual, auditory, hands-on)
- Difficulty level (easy, medium, hard)

If a field is not mentioned, return "Not mentioned".

Text: "{text}"

Return the result in this format with apropriate string and integer values:
{{
    "Goal": "Exam"/"Project"/"Mastery"
    "Subject": ...
    "Study Time in Days": ...
    "Study Time in Hours": ...
    "Current Grade/Year": ... "Grade"/"Year"
    "Learning Style": ... "Visual"/"Auditory"/"Hands-on"
    "Difficulty Level": ...
}}
"""
    response = client.models.generate_content(model="gemini-2.5-flash",contents=prompt)
    return response.text



# # Example Classification
# predicted_label = classify_text("I want to study for my upcoming exam on machine learning.")
# print(f"Predicted label: {predicted_label[0]}, Confidence: {predicted_label[1]:.4f}")

# Streamlit UI

st.title("Studbud: Study Planner")
st.write("Enter your study-related query below:")
user_input = st.text_area("Input", height=200, label_visibility="collapsed")

button_1 = st.button("Classify Text")
button_2 = st.button("Reload")

if button_1 and user_input.strip():
    
    with st.spinner("Extracting..."):
        label, confidence = classify_text(user_input)
        response = extract_study_info(user_input)

    try:
        # Attempt to parse the response as a dictionary
        extracted_info = eval(response.replace("```","").replace("json","").strip())  # Convert response to dictionary
        st.session_state.extracted_info = extracted_info
        st.session_state.label = label
        st.session_state.show_form = True  # Trigger form rendering
        st.success(f"Scenario: {label}.")
        st.write("Scenario is wrong? Reload to try again. Try to be more specific in your text input.")


        # with st.form("study_plan_form"):
        #     goal = st.selectbox("🎯 Goal", options=["Exam", "Project", "Mastery"], index=["Exam", "Project", "Mastery"].index(label))
        #     subject = st.text_input("📘 Subject", value=extracted_info["Subject"])
        #     study_hours = st.number_input("🕒 Study Time per Day (hrs)", value=extracted_info["Study Time in Days"], min_value=0)
        #     study_days = st.number_input("📅 Study Duration (days)", value=extracted_info["Study Time in Hours"], min_value=1)
        #     grade = st.text_input("🎓 Current Grade", value=extracted_info["Current Grade/Year"])
        #     learning_style = st.selectbox("🧠 Learning Style", options=["Visual", "Auditory", "Hands-on"], index=["Visual", "Auditory", "Hands-on"].index(extracted_info["Learning Style"]))
        #     difficulty = st.text_input("⚠️ Difficulty Areas", value=extracted_info["Difficulty Level"])
        #     submit = st.form_submit_button("✅ Confirm & Generate Study Plan")
        #     if submit:
        #         st.write("Generating study plan...")
        #         # Here you would typically call a function to generate the study plan based on the inputs
        #         # For now, we will just display the inputs as a placeholder
        #         st.write("Study Plan:")
        #         st.write(f"Goal: {goal}")
        #         st.write(f"Subject: {subject}")
        #         st.write(f"Study Time per Day: {study_hours} hrs")
        #         st.write(f"Study Duration: {study_days} days")
        #         st.write(f"Current Grade: {grade}")
        #         st.write(f"Learning Style: {learning_style}")
        #         st.write(f"Difficulty Areas: {difficulty}")

        #         print(f"Goal: {goal}, Subject: {subject}, Study Time per Day: {study_hours} hrs, Study Duration: {study_days} days, Current Grade: {grade}, Learning Style: {learning_style}, Difficulty Areas: {difficulty}")
        #         st.success("Study plan generated successfully!")

    except Exception as e:
        st.error(f"Error parsing response: {e}")
        st.write("Response was:", response)
elif button_1 :
    st.error("Please enter some text to classify.")

# Reload the model
if button_2:
    tokenizer, model = load_model()
    st.success("Model reloaded successfully!")


    
if st.session_state.get("show_form"):
    extracted_info = st.session_state.extracted_info
    label = st.session_state.label

    # Override the goal label if it does not match the extracted info
    if extracted_info["Goal"] != label:
        new_label = extracted_info["Goal"]
        st.warning(f"Detected goal '{new_label}' does not match classified label '{label}'. Using detected goal instead.")

    with st.form("study_plan_form"):
        goal = st.selectbox("🎯 Goal (Edit If Necessary)", ["Exam", "Project", "Mastery"], index=["Exam", "Project", "Mastery"].index(new_label))
        subject = st.text_input("📘 Subject", value=extracted_info["Subject"])
        study_hours = st.number_input("🕒 Study Time per Day (hrs)", value=int(extracted_info["Study Time in Days"]), min_value=0)
        study_days = st.number_input("📅 Study Duration (days)", value=int(extracted_info["Study Time in Hours"]), min_value=1)
        grade = st.text_input("🎓 Current Grade", value=extracted_info["Current Grade/Year"])
        learning_style = st.selectbox("🧠 Learning Style", ["Visual", "Auditory", "Hands-on"], index=["Visual", "Auditory", "Hands-on"].index(extracted_info["Learning Style"]))
        difficulty = st.text_input("⚠️ Difficulty Areas", value=extracted_info["Difficulty Level"])
        submit = st.form_submit_button("✅ Confirm & Generate Study Plan")

    if submit:
        st.write("### ✅ Study Plan Generated")
        st.write(f"Goal: {goal}")
        st.write(f"Subject: {subject}")
        st.write(f"Study Time per Day: {study_hours} hrs")
        st.write(f"Study Duration: {study_days} days")
        st.write(f"Current Grade: {grade}")
        st.write(f"Learning Style: {learning_style}")
        st.write(f"Difficulty Areas: {difficulty}")
        st.success("🎉 Study plan ready!")
