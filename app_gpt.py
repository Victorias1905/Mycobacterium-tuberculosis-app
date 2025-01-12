import streamlit as st
import openai
from pymilvus import connections, Collection, MilvusClient
from pdfminer.high_level import extract_text
import unicodedata
import tiktoken
import os
import subprocess
import tempfile

# Streamlit configuration
st.set_page_config(layout="wide")

# Initialize session state variables
if "debug_log" not in st.session_state:
    st.session_state.debug_log = []
if "answers" not in st.session_state:
    st.session_state.answers = []

api_key = st.secrets["general"]["OPENAI_API_KEY"]
zilliz_uri = "https://in03-03d63efede22046.serverless.gcp-us-west1.cloud.zilliz.com"
zilliz_token = st.secrets["general"]["zilliz_token"]
collection_name = "Mycobacterium"
embedding_field = "vector"

# Model selection
model_ids = ["o1-2024-12-17","gpt-4o-2024-08-06","ft:gpt-4o-mini-2024-07-18:mtbc-project::Akwtgx7I", 
             "ft:gpt-4o-mini-2024-07-18:mtbc-project::AkyZCr4h", 
             "gpt-4o-mini-2024-07-18"]
model = st.selectbox("Select Model", model_ids)

client = openai.OpenAI(api_key=api_key)
client_milvus = MilvusClient(uri=zilliz_uri, token=zilliz_token)

# Connect to Zilliz Cloud
try:
    connections.connect("default", uri=zilliz_uri, token=zilliz_token)
    zilliz_client = MilvusClient(uri=zilliz_uri, token=zilliz_token)
    collection = Collection(collection_name)
except Exception as e:
    st.write(f"Failed to connect to Zilliz Cloud: {e}")

tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

def get_embedding(user_query):
    """Generate embedding for text using OpenAI."""
    try:
        embedding_response = openai.OpenAI(api_key=api_key).embeddings.create(
            input=user_query,
            model="text-embedding-3-small"
        )
        return embedding_response.data[0].embedding
    except Exception as e:
        st.write(f"Error generating embedding: {e}")
        return None

def query_zilliz(user_vector, top_k=3):
    """Query Zilliz database."""
    try:
        closest_results = collection.search(
            data=[user_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 1024}},
            limit=top_k,
            output_fields=["text", "metadata"]
        )
        return closest_results
    except Exception as e:
        st.write(f"Error querying Zilliz: {e}")
        return None

def extract_relevant_data(closest_results):
    retrieved_texts = []
    for hits in closest_results:
        for hit in hits:
            text = hit.text
            metadata = hit.metadata
            retrieved_texts.append({"text": text, "metadata": metadata})
    return retrieved_texts

def construct_prompt_with_references(user_query, retrieved_texts):
    references_str = "\n\n".join(
        [f"Text: {data['text']}\nMetadata: {data['metadata']}"
         for data in retrieved_texts]
    )
    return (
        f"User query: {user_query}\n\n"
        f"Retrieved references:\n{references_str}\n\n"
        "Provide a response for the query. The response should be based on the references."
    )

def get_response(prompt, retrieved_texts):
    """Generate a response using OpenAI."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()
        full_response = {
            "answer": answer,
            "metadata": [data["metadata"] for data in retrieved_texts]
        }
        return full_response
    except Exception as e:
        st.write(f"Error generating response: {e}")
        return None

# -------------
# Main App
# -------------

st.title("OpenAI models + vector database")

user_query = st.text_input("Enter your query:")
if user_query:
    user_vector = get_embedding(user_query)
    if user_vector:
        closest_results = query_zilliz(user_vector, top_k=3)
        if closest_results:
            retrieved_texts = extract_relevant_data(closest_results)
            if retrieved_texts:
                prompt = construct_prompt_with_references(user_query, retrieved_texts)
                response = get_response(prompt, retrieved_texts)
                if response:
                    # Store response to session_state
                    st.session_state.answers.append({
                        "question": user_query,
                        "answer": response["answer"],
                        "metadata": response["metadata"]
                    })
            else:
                st.write("No relevant references found.")
        else:
            st.write("No results found in Zilliz.")
    else:
        st.write("Failed to generate embedding.")

# Display all Q&A history from session state
st.write("## Q&A History")
for idx, item in enumerate(st.session_state.answers, start=1):
    st.write(f"**Q{idx}:** {item['question']}")
    st.write(f"**Answer:** {item['answer']}")
    st.write("**Metadata:**")
    for metadata in item["metadata"]:
        st.write(f"- Reference: {metadata.get('reference', 'No reference found')}")
# -------------
# Update Database Section
# -------------

st.title("Update Database")

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
preprocess_button = st.button("Preprocess", key="Preprocess")

def process_pdf(pdf_file):
    if hasattr(pdf_file, 'name') and pdf_file.name:
        reference_name = os.path.basename(pdf_file.name)
    elif hasattr(pdf_file, 'filename') and pdf_file.filename:
        reference_name = os.path.basename(pdf_file.filename)
    else:
        reference_name = "uploaded_file.pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name
    pdf_text = extract_text(tmp_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    lines = pdf_text.splitlines()
    keywords = ['ACKNOWLEDGMENTS', 'Acknowledgments', 'References', 'REFERENCES']
    end_index = None
    for i, line in enumerate(lines):
        if any(keyword in line for keyword in keywords):
            end_index = i
            break
    content_to_keep = lines[:end_index] if end_index is not None else lines
    cleaned_text = "\n".join(content_to_keep)
    return cleaned_text, reference_name

def chunk_documents_with_references(update_list):
    chunks = []
    chunk_metadata = []
    for item in update_list:
        tok = tiktoken.encoding_for_model("text-embedding-3-small")
        tokens = tok.encode(item.get("text"))
        max_tokens = 7000
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = tok.decode(chunk_tokens)
            chunks.append(chunk_text)
            chunk_metadata.append({"filename": item.get("filename")})
    return chunks, chunk_metadata

def generate_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        embeddings.append(response)
    return embeddings

if preprocess_button:
    if uploaded_files:
        update_list = []
        for pdf_file in uploaded_files:
            cleaned_text, reference_name = process_pdf(pdf_file)
            output = {
                "filename": reference_name,
                "text": cleaned_text
            }
            update_list.append(output)
        st.success("Files processed successfully!")
        
        chunks, chunk_metadata = chunk_documents_with_references(update_list)
        st.write("Number of chunks created:", len(chunks))

        embeddings = generate_embeddings(chunks)
        vectors = [embedding.data[0] for embedding in embeddings]
        vectors_float = [vector.embedding for vector in vectors]
        ids = list(range(len(chunks)))  # unique IDs

        for id_, vector, chunk, meta in zip(ids, vectors_float, chunks, chunk_metadata):
            data_to_insert = {
                "id": id_,
                "vector": vector,
                "text": chunk,
                "metadata": meta,
            }
            client_milvus.insert(collection_name="Mycobacterium", data=data_to_insert)

        st.write("Data inserted successfully!")
    else:
        st.warning("Please upload at least one PDF file.")















