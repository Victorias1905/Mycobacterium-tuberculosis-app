import openai
import streamlit as st
from pymilvus import connections, Collection, MilvusClient
from pdfminer.high_level import extract_text

import unicodedata
import tiktoken
import os
import subprocess
import tempfile
# Streamlit configuration
st.set_page_config(layout="wide")
api_key = st.secrets["general"]["OPENAI_API_KEY"]

# Zilliz Cloud connection details
zilliz_uri = "https://in03-03d63efede22046.serverless.gcp-us-west1.cloud.zilliz.com"
zilliz_token = "641b977aa113eb7f095c50a472347f9f089f6ee89e1346d3ea316db3223c8cf9b4f42bfd705ccb1fad8d7b00d62f1b27bfe8a59e"
collection_name = "Mycobacterium"
embedding_field = "vector"  # Field name for embeddings in Zilliz
model="ft:gpt-4o-mini-2024-07-18:mtbc-project::Akwtgx7I"
# Connect to Zilliz Cloud
try:
    connections.connect("default", uri=zilliz_uri, token=zilliz_token)
    zilliz_client = MilvusClient(uri=zilliz_uri, token=zilliz_token)
    collection = Collection(collection_name)
   
except Exception as e:
    st.write(f"Failed to connect to Zilliz Cloud: {e}")



def get_embedding(user_query):
    """Generate embedding for text using OpenAI."""
    try:
        
        embedding_response = openai.OpenAI(api_key=api_key).embeddings.create(
            input=user_query,
            model="text-embedding-3-small"
        )
        user_vector = embedding_response.data[0].embedding
        return user_vector
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
        
        return  closest_results
    except Exception as e:
        st.write(f"Error querying Zilliz: {e}")
        return None
def extract_relevant_data(closest_results):
    
    retrieved_texts = []
    for hits in closest_results:      # For each query
        for hit in hits:             # For each hit within that query
            text = hit.text
            metadata = hit.metadata
            retrieved_texts.append({"text": text, "metadata": metadata})
    st.write(len(retrieved_texts))   
    return retrieved_texts
def construct_prompt_with_references(user_query, retrieved_texts):
     references_str = "\n\n".join(
        [f"Text: {data['text']}\nMetadata: {data['metadata']}" for data in retrieved_texts]
    )
     prompt = (
        f"User query: {user_query}\n\n"
        f"Retrieved references:\n{references_str}\n\n"
        "Provide response for the query. The responce should be based on the references."
    )
  
     return prompt
def get_response(prompt, retrieved_texts):
    """Generate a response using OpenAI."""
    try:
       
        client = openai.OpenAI(api_key=api_key)
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

# Streamlit App
st.title("Debugging Zilliz Retrieval")

user_query = st.text_input("Enter your query:")

if user_query:
    # Step 1: Generate embedding
    user_vector = get_embedding(user_query)
    if user_vector:
        closest_results = query_zilliz(user_vector, top_k=3)

        if closest_results:
            # Step 3: Extract relevant data
            retrieved_texts = extract_relevant_data(closest_results)

            if retrieved_texts:
                # Step 4: Construct prompt with references
                prompt = construct_prompt_with_references(user_query, retrieved_texts)

                # Step 5: Get response from OpenAI
                response = get_response(prompt, retrieved_texts)
                if response:
                    st.write("Answer:")
                    st.write(response["answer"])
                    for metadata in response["metadata"]:
                        st.write(f"- Reference: {metadata.get('reference', 'No reference found')}")
            else:
                st.write("No relevant references found.")
        else:
            st.write("No results found in Zilliz.")
    else:
        st.write("Failed to generate embedding.")
st.title("Update Database")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
preprocess_button=st.button("Preprocess", key="Preprocess")
process_button = st.button('Update', key='Update')


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
    
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small") 
def chunk_documents_with_references(Update_list, max_tokens=7000):
    """Split documents into chunks based on token limit and add references."""
    chunks = []
    chunk_metadata = []
    for item in Update_list:
        st.write(item.get("text"))
        tokens = tokenizer.encode(item.get("text"))
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = tokenizer.decode(chunk_tokens)  # Convert tokens back to text
            chunks.append(chunk_text)
            chunk_metadata.append({"filename": item.get("filename")})
    
    return chunks, chunk_metadata

def generate_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
        input=chunk,
        model="text-embedding-3-small")
        embeddings.append(response)
    
    return embeddings
    
if preprocess_button:
    if uploaded_files:
        Update_list=[]
        for pdf_file in uploaded_files:
            cleaned_text, reference_name=process_pdf(pdf_file)
            output = {
                "filename": reference_name,
                "text": cleaned_text }
            Update_list.append(output)
        st.success("Files processed successfully!")
        chunks, chunk_metadata = chunk_documents_with_references(Update_list, tokenizer)
        
        st.write("Number of chunks created:", len(chunks))
        st.write("Chunk metadata example:", chunk_metadata[:3])  # show first few metadata entries
        embeddings=generate_embeddings(chunks)
        vectors = [embedding.data[0] for embedding in embeddings] 
        vectors_float = [vector.embedding for vector in vectors]
        client = MilvusClient(uri="https://in03-03d63efede22046.serverless.gcp-us-west1.cloud.zilliz.com",
                              token="641b977aa113eb7f095c50a472347f9f089f6ee89e1346d3ea316db3223c8cf9b4f42bfd705ccb1fad8d7b00d62f1b27bfe8a59e")

        # Prepare data for insertion
        ids = list(range(len(chunks)))  # Generate unique IDs
        for id,vector, chunk, metadata in zip(ids,vectors_float, chunks, chunk_metadata):
            data_to_insert = {
                "id": id,
                "vector": vector,
                "text": chunk,
                "metadata": metadata,  
            }
        
            client.insert(collection_name="Mycobacterium", data=data_to_insert)
        print("Data inserted successfully!")
    else:
        st.warning("Please upload at least one PDF file.")











