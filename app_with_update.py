
import openai
import streamlit as st
import json
from pdfminer.high_level import extract_text
from transformers import AutoTokenizer, AutoModelForCausalLM
import unicodedata
import tiktoken
import os

# Streamlit secrets for the OpenAI API key
api_key = st.secrets["general"]["OPENAI_API_KEY"]

# Function to load the latest model name
def load_latest_model():
    try:
        file_path = os.path.join(os.getcwd(), "latest_model.json")
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file).get("model_name")
    except Exception as e:
        st.error(f"Failed to load the latest model. Error: {e}")
        return None

# Function to save the latest model name
def save_latest_model(model_name):
    try:
        file_path = os.path.join(os.getcwd(), "latest_model.json")
        st.write(f"Saving model to: {file_path}")  # Debugging log
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump({"model_name": model_name}, file, indent=4)
        # Verify the content after saving
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            st.write(f"Updated file content: {content}")
        st.success(f"Model name successfully saved to {file_path}")
    except Exception as e:
        st.error(f"Failed to save the model name. Error: {e}")

# Load the current model name
model_name = load_latest_model()
if model_name is None:
    st.error("Model name could not be loaded. Please check 'latest_model.json'.")

# OpenAI client
client = openai.OpenAI(api_key=api_key)

# Function to get a response from the OpenAI model
def get_response(prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Streamlit app title
st.title("GPT-MTBC gpt-4o-mini")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input and response generation
user_input = st.text_input("You:", key="user_input")
if user_input:
    response = get_response(user_input)
    st.session_state.chat_history.append({"user": user_input, "bot": response})

# Display chat history
for chat in st.session_state.chat_history:
    st.write(f"You: {chat['user']}")
    st.write(f"Bot: {chat['bot']}")

# Tokenization setup
encoding = tiktoken.encoding_for_model("gpt-4o-mini-2024-07-18")

# Define preprocessing function (unchanged)
def pdf_preprocessing(article, outfile):
    # Existing implementation of preprocessing
    pass

# Streamlit app for model update
st.title("Model update - upload at least ten articles")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
preprocess_button = st.button("Preprocess", key="Preprocess")
process_button = st.button("Update", key="Update")

# Preprocess uploaded files
if preprocess_button:
    if uploaded_files:
        with open("json_output_file", "w", encoding="utf-8") as outfile:
            for article in uploaded_files:
                pdf_preprocessing(article, outfile)
        st.success("Files processed successfully!")

# Fine-tune model
if process_button:
    response = client.files.create(
        file=open("json_output_file", "rb"),
        purpose="fine-tune"
    )
    file_id = response.id
    fine_tune_response = client.fine_tuning.jobs.create(
        model=model_name,
        training_file=file_id,
        hyperparameters={"n_epochs": 2, "learning_rate_multiplier": 1, "batch_size": 1}
    )
    fine_tune_job_id = fine_tune_response.id

    while True:
        job_response = client.fine_tuning.jobs.retrieve(fine_tune_job_id)
        job_status = job_response.status
        if job_status == "succeeded":
            model_name = job_response.fine_tuned_model
            if model_name:
                save_latest_model(model_name)
            break
        elif job_status in ["failed", "cancelled"]:
            st.error(f"Fine-tuning failed or was cancelled. Status: {job_status}")
            break
        else:
            st.info(f"Fine-tuning in progress. Current status: {job_status}")

           
   
   
