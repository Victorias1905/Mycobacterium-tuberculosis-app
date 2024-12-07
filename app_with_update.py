
import openai
import streamlit as st
import json
from pdfminer.high_level import extract_text
from transformers import AutoTokenizer, AutoModelForCausalLM
import unicodedata
import tiktoken
import os

api_key = st.secrets["general"]["OPENAI_API_KEY"]
model_name="latest_model.json"
LATEST_MODEL_FILE = "latest_model.json"
with open(LATEST_MODEL_FILE, "r") as file:
    model_name = json.load(file).get("model_name")
def save_latest_model(model_name):
    with open(LATEST_MODEL_FILE, "w") as file:
        json.dump({"model_name": model_name}, file)

def save_latest_model(model_name):
    with open(LATEST_MODEL_FILE, "w") as file:
        json.dump({"model_name": model_name}, file)

client = openai.OpenAI(api_key=api_key)
def get_response(prompt):
    response = client.chat.completions.create(
    model=model_name,  
    messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip()

st.title("GPT-MTBC gpt-4o-mini")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("You:", key="user_input")
if user_input:
    response = get_response(user_input)
    st.session_state.chat_history.append({"user": user_input, "bot": response})

for chat in st.session_state.chat_history:
    st.write(f"You: {chat['user']}")
    st.write(f"Bot: {chat['bot']}")
# Set up encoding for tokenization
encoding = tiktoken.encoding_for_model("gpt-4o-mini-2024-07-18")

# Define the preprocessing function
def pdf_preprocessing(article, outfile):
    # Extract text from the PDF and save it to a temporary file
    with open("new_material", 'w', encoding='utf-8') as file:
        file.write(extract_text(article))

    # Read the text and process it
    with open("new_material", 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Find the index to truncate content after specified keywords
    end_index = None
    for i, line in enumerate(lines):
        if any(keyword in line for keyword in ['ACKNOWLEDGMENTS', 'Acknowledgments', 'References', 'REFERENCES']):
            end_index = i
            break

    # Keep content up to the found index
    content_to_keep = lines[:end_index] if end_index is not None else lines

    # Save the cleaned text
    with open("new_material", 'w', encoding='utf-8') as file:
        file.writelines(content_to_keep)

    # Load the cleaned text and prepare the JSON structure
    with open("new_material", 'r', encoding='utf-8') as file:
        text = file.read()
        data = {"text": text}
        with open("new_material", 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)

    # Process the JSON to create message chunks
    with open("new_material", 'r', encoding='utf-8') as infile:
        data = json.load(infile)

        if "text" in data:
            text = data["text"].strip()
            if text:
                text = unicodedata.normalize("NFKD", text)

            text_lines = text.splitlines()
            header_lines = text_lines[:20]
            header = "\n".join(header_lines).strip() + "\n"
            main_content = "\n".join(text_lines[20:]).strip()

            # Tokenize the main content
            tokens = encoding.encode(main_content)
            number_of_tokens = len(tokens)
            header_tokens = len(encoding.encode(header))

            # Create chunks based on token limits
            if number_of_tokens + header_tokens >= 20000:
                max_tokens_per_chunk = 20000 - header_tokens
                chunks = [tokens[i:i + max_tokens_per_chunk] for i in range(0, len(tokens), max_tokens_per_chunk)]
            else:
                chunks = [tokens]

            # Write formatted chunks to the output file
            for i, chunk in enumerate(chunks):
                chunk_text = encoding.decode(chunk)
                full_text = header + chunk_text
                formatted_chunk = {
                    "messages": [
                        {"role": "system", "content": "You are a highly knowledgeable assistant on MTBC. You always provide answers with references."},
                        {"role": "user", "content": "Please provide an answer on the following topic, supporting your response with relevant references."},
                        {"role": "assistant", "content": full_text}
                    ]
                }
                outfile.write(json.dumps(formatted_chunk, ensure_ascii=False) + "\n")
        else:
            raise ValueError("Key 'text' not found in JSON file.")
    return outfile


# Streamlit app
st.title("Model update - upload at least ten articles")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Process each uploaded file
if uploaded_files:
    with open("json_output_file", 'a', encoding='utf-8') as outfile:
        for article in uploaded_files:
            pdf_preprocessing(article, outfile)

    st.success("Files processed successfully!")


    response = client.files.create(
    file=open("json_output_file", 'rb'),
    purpose='fine-tune'
    )
    file_id = response.id
    training_files = []
    training_files.append(file_id)
    for training_file_id in training_files:
 
        fine_tune_response = client.fine_tuning.jobs.create(
            model=model_name,  
            training_file=training_file_id,
            hyperparameters={"n_epochs":3, 
                             "learning_rate_multiplier":1,
                            "batch_size":1
            }
        )
        fine_tune_job_id = fine_tune_response.id
        job_response = client.fine_tuning.jobs.retrieve(fine_tune_job_id)
       
            
            
             
    print("All fine-tuning jobs processed.")
    model_name=job_response.fine_tuned_model
    save_latest_model(model_name)
