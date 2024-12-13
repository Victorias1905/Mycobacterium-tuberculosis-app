
import openai
import streamlit as st
import json
from pdfminer.high_level import extract_text
from transformers import AutoTokenizer, AutoModelForCausalLM
import unicodedata
import tiktoken
import os
import subprocess

api_key = st.secrets["general"]["OPENAI_API_KEY"]

with open("latest_model.json", "r") as file:
    model_name = json.load(file).get("model_name")
def save_latest_model(model_name):
    try:
        # Get the absolute path of the JSON file
        file_path = os.path.abspath("latest_model.json")
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump({"model_name": model_name}, file)
        st.success(f"Model name successfully saved to {file_path}")
    except Exception as e:
        st.error(f"Failed to save the model name. Error: {e}")

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
            if number_of_tokens + header_tokens >= 4000:
                max_tokens_per_chunk = 4000 - header_tokens
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
preprocess_button=st.button("Preprocess", key="Preprocess")
process_button = st.button('Update', key='Update')

# Process each uploaded file
if preprocess_button:
    if uploaded_files:
        with open("json_output_file", 'w', encoding='utf-8') as outfile:
            for article in uploaded_files:
                pdf_preprocessing(article, outfile)
    
        st.success("Files processed successfully!")

if process_button:
    response = client.files.create(
    file=open("json_output_file", 'rb'),
    purpose='fine-tune'
    )
    file_id = response.id
    fine_tune_response = client.fine_tuning.jobs.create(
        model=model_name,  
        training_file=file_id,
        hyperparameters={"n_epochs":2, 
                         "learning_rate_multiplier":1,
                        "batch_size":1} )
    fine_tune_job_id = fine_tune_response.id
 
    while True:
        job_response = client.fine_tuning.jobs.retrieve(fine_tune_job_id)
        job_status = job_response.status
        if job_status == "succeeded":
            model_name = job_response.fine_tuned_model
            if model_name:
                save_latest_model(model_name)
            break



def push_to_git():
    try:
        # Ensure the current working directory is within a Git repo
        repo_path = os.path.abspath(".")
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_path, check=True, capture_output=True, text=True)

        # Configure Git user
        subprocess.run(["git", "config", "user.name", "Victorias1905"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "102805197+Victorias1905@users.noreply.github.com"], cwd=repo_path, check=True)

        # Add and commit changes
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True)
        try:
            subprocess.run(["git", "commit", "-m", f"Update model name to {model_name}"], cwd=repo_path, check=True)
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8") if e.stderr else "No error message"
            if "nothing to commit" in stderr:
                st.warning("No changes to commit. The repository is up-to-date.")
                return
            else:
                st.error(f"Git commit failed: {stderr}")
                return

        # Authenticate and set remote URL
        token = st.secrets["general"]["GITHUB_TOKEN"]
        repo_name = "Mycobacterium-tuberculosis-app"
        username = "Victorias1905"
        auth_remote = f"https://{token}@github.com/{username}/{repo_name}.git"
        subprocess.run(["git", "remote", "set-url", "origin", auth_remote], cwd=repo_path, check=True)

        # Push changes to the remote repository
        result = subprocess.run(["git", "push", "origin", "main"], cwd=repo_path, capture_output=True, text=True)
        stdout = result.stdout or "No output"
        stderr = result.stderr or "No error message"

        if result.returncode != 0:
            st.error(f"Git push failed:\nstdout: {stdout}\nstderr: {stderr}")
        else:
            st.success("Changes successfully pushed to GitHub!")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8") if e.stderr else "No error message"
        stdout = e.stdout.decode("utf-8") if e.stdout else "No output"
        st.error(f"Git command failed:\nstdout: {stdout}\nstderr: {stderr}")
    except Exception as e:
        st.error(f"Unexpected error occurred: {str(e)}")





if st.button("Push to GitHub"):
    push_to_git()  
