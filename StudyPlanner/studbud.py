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

model_path = "psuedopadel24/bert-study-planner"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path,use_auth_token=HF_API_KEY)
    model = AutoModelForSequenceClassification.from_pretrained(model_path,use_auth_token=HF_API_KEY)
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
- Study time available in hours (Return int value, only maximum one)
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
    "Study Time in Hours": ... (Per day)
    "Current Grade/Year": ... "Grade"/"Year"
    "Learning Style": [...] (Example: ["Visual"/"Auditory"/"Hands-on"] NOT A DICTIONARY!!!) 
    "Difficulty Level": "Hard"/"Medium"/"Easy"
}}
"""
    response = client.models.generate_content(model="gemini-2.5-flash",contents=prompt)
    return response.text

def generate_study_plan(goal, subject, study_hours, study_days, grade, learning_style, difficulty):
    plan = ""

    # Intro 
    plan += f"🎓 **Personalized Study Plan for {subject} ({goal})**.  \n  \n"
    plan += f"Student Level: {grade}.  \n"
    plan += f"Study Duration: {study_days} days for {study_hours} hrs/day.  \n"
    plan += f"Preferred Learning Style(s): {', '.join(learning_style)}.  \n"
    plan += f"Difficulty Level: {difficulty}.  \n  \n"

    # Scenario-specific strategy
    if goal == "Exam":
        plan += "📌 **Strategy:**\n"
        plan += "- Prioritize topics you're weak in.\n"
        plan += "- Use spaced repetition and timed mock tests.\n"
        plan += "- Increase review frequency as exam approaches.\n\n"
    elif goal == "Project":
        plan += "📌 **Strategy:**\n"
        plan += "- Break down the project into smaller subtasks: research, design, implementation, testing.\n"
        plan += "- Allocate time per milestone and review regularly.\n\n"
    elif goal == "Mastery":
        plan += "📌 **Strategy:**\n"
        plan += "- Balance theory and practice.\n"
        plan += "- Mix media: books, videos, hands-on experiments.\n"
        plan += "- Weekly reflections to track deep understanding.\n\n"

    # Daily Plan
    plan += "📅 **Daily Plan Overview:**\n"
    for day in range(1, study_days + 1):
        plan += f"- **Day {day}:**\n"
        if goal == "Exam":
            plan += f"  - {subject}: Study weak topics for {study_hours / 2} hrs,  \nsolve practice problems for {study_hours / 2} hrs.\n"
        elif goal == "Project":
            plan += f"  - {subject}: Progress in assigned task.  \nLog findings and prepare for next phase.\n"
        elif goal == "Mastery":
            plan += f"  - {subject}: Split time into {', '.join(learning_style)} based sessions (each ~{study_hours // len(learning_style)} hrs).\n"
    
    
    plan += "\n**Tips:**\n"
    # Gemini Insight
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""Provide daily study tips in short for a plan: {plan} in {subject} for a student with {grade} grade with {difficulty.lower()} difficulty level. 
        NOTE: This is being used in a Streamlit Web App, so keep the response short and concies without anything extra at the start or end, it should be in markdown points with - at the start. 
        But with some motivating emojis at the start of each day point and proper linebreak after each day point \n\n"""
    )

    plan += f"  \n{response.text.strip()}\n"

    # Closing
    plan += "\n✅  Good Luck!  \n"

    return plan


# # Example Classification
# predicted_label = classify_text("I want to study for my upcoming exam on machine learning.")
# print(f"Predicted label: {predicted_label[0]}, Confidence: {predicted_label[1]:.4f}")

# Streamlit UI

st.markdown("""
            <style>
                div[data-testid="stColumn"] {
                    width: fit-content !important;
                    flex: unset;
                }
                div[data-testid="stColumn"] * {
                    width: fit-content !important;
                }
            </style>
            """, unsafe_allow_html=True)

st.title("🎓 Studbud: Study Planner")
st.write("Enter your study-related query below:")
user_input = st.text_area("Input", height=200,placeholder=f"""Mention details like: Goal, Subject, Time in days, Time in hours per day, Current grade, Learning Style and Current difficulty.""", label_visibility="collapsed")


# button_1 = st.button("Generate Study Form")
# button_2 = st.button("Reload")

col1, col2 = st.columns([1,1])

with col1:
    button_1 = st.button("Generate Study Form", key="button_1")
with col2:
    button_2 = st.button("Reload Model", key="button_2")



if button_1 and user_input.strip():
    
    with st.spinner("Extracting..."):
        label, confidence = classify_text(user_input)
        response = extract_study_info(user_input)

    try:
        # Attempt to parse the response as a dictionary
        extracted_info = eval(response.replace("```","").replace("json","").strip())  # Convert response to dictionary
        #print(f"Extracted info:\n {extracted_info}")  # Debugging line to check the extracted info
        st.session_state.extracted_info = extracted_info
        st.session_state.label = label
        st.session_state.show_form = True  # Trigger form rendering
        st.write("Scenario is wrong? Just change the scenario in the form.")

    except Exception as e:
        st.session_state.show_form = False  # Reset form state
        st.error(f"Error parsing response: {e}")
        st.write("Response was:", response)

elif button_1 :
    st.error("Please enter some text to classify.")

# Reload the model
if button_2:
    tokenizer, model = load_model()
    st.success("Model reloaded successfully!")


# Preserve form state in reruns
if st.session_state.get("show_form"):
    
    extracted_info = st.session_state.extracted_info
    label = st.session_state.label

    # Handle missing fields in the extracted info
    for key, value in extracted_info.items():
        if value == "Not mentioned":
            if key == "Subject" or key == "Current Grade/Year":
                extracted_info[key] = ""
            if key == "Learning Style":
                extracted_info[key] = []
            if key == "Difficulty Level":
                extracted_info[key] = "Medium"
            if key == "Study Time in Days" or key == "Study Time in Hours":
                extracted_info[key] = 0

    # Handle Case where no scenario is given
    if extracted_info["Goal"] == "Not mentioned":
        
        st.warning(f"There is no scenario detected. Please enter a scenario.")

    label = "Not mentioned" if label == "Not mentioned" else label

    # Override the goal label if it does not match the extracted info
    if extracted_info["Goal"] != label:
        #st.warning(f"Detected goal '{extracted_info["Goal"]}' does not match classified label '{label}'. Using detected goal instead.")
        label = extracted_info["Goal"]  
 

    st.success(f"Scenario: {label}.")

    st.markdown("### ✏️ Fill Out Study Preferences")

    try: 
        with st.form("study_plan_form"):

            # Handle Case where no scenario is given
            options = ["Exam", "Project", "Mastery"]
            index = options.index(label) if label in options else None
            goal = st.selectbox("🎯 Goal (Edit If Necessary)", options, index=index)
            subject = st.text_input("📘 Subject", value=extracted_info["Subject"])
            study_hours = st.number_input("🕒 Study Time per Day (hrs)", value=int(extracted_info["Study Time in Hours"]))
            study_days = st.number_input("📅 Study Duration (days)", value=int(extracted_info["Study Time in Days"]))
            grade = st.text_input("🎓 Current Grade", value=extracted_info["Current Grade/Year"])
            learning_style = st.multiselect("🧠 Learning Style", ["Visual", "Auditory", "Hands-on"], default = extracted_info["Learning Style"])
            difficulty = st.selectbox("⚠️ Difficulty Level",["Hard","Medium","Easy"], index=["Hard","Medium","Easy"].index(extracted_info["Difficulty Level"]))
        
            submit = st.form_submit_button("✅ Confirm & Generate Study Plan")


    except Exception as e:
        st.warning(f"Error in form input: {e}. Please check your inputs.")
        st.stop()

   

    if submit:
        # Validate inputs
        missing_fields = []
        if not subject or subject.lower() == "not mentioned":
            missing_fields.append("Subject")
        if not grade or grade.lower() == "not mentioned":
            missing_fields.append("Current Grade/Year")
        if not difficulty or difficulty.lower() == "not mentioned":
            missing_fields.append("Difficulty Level")
        if study_days == 0 or  study_days < 1 or study_days > 365:
            missing_fields.append("Study Time in Days")
        if study_hours == 0 or study_hours < 0 or study_hours > 24:
            missing_fields.append("Study Time in Hours")

        if missing_fields:
            st.warning(f"The following fields were not correctly filled: {', '.join(missing_fields)}. You can edit them manually in the form.")

        # If all fields are valid, generate the study plan
        else:
            with st.spinner("Generating study plan..."):
                st.markdown("### 📅 Study Plan")

                plan = generate_study_plan(
                    goal=goal,
                    subject=subject,
                    study_hours=study_hours,
                    study_days=study_days,
                    grade=grade,
                    learning_style=learning_style,
                    difficulty=difficulty
                )
                st.write(plan)
            # Display the generated study plan
            
            st.success("🎉 Study plan ready!")
            
            
            
