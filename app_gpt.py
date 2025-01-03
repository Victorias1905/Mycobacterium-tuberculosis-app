import openai
import streamlit as st
from pymilvus import connections, Collection, MilvusClient

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
    st.write("Connected to Zilliz Cloud successfully!")
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

def query_zilliz(user_vector, top_k=5):
    """Query Zilliz database."""
    try:
        
        closest_results = collection.search(
            data=[user_vector],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 5}},
            limit=top_k,
            output_fields=["text", "metadata"] 
        )
      
        return  closest_results
    except Exception as e:
        st.write(f"Error querying Zilliz: {e}")
        return None
def extract_relevant_data( closest_results):
    """Extract text and metadata from Zilliz search results."""
    retrieved_texts = []
    for result in  closest_results:
        # Extract text and metadata fields
        text = result.text
        metadata = result.metadata
        # Combine into a single string for the prompt
        retrieved_texts.append(f"Text: {text}\nMetadata: {metadata}")
    return retrieved_texts
def construct_prompt_with_references(user_query, retrieved_texts):
    references_str = "\n\n".join(retrieved_texts)
    prompt = (
        f"User query: {user_query}\n\n"
        f"Retrieved references:\n{references_str}\n\n"
        "Provide the most relevant response based on the query and references."
    )
    st.write(prompt)
    return prompt
def get_response(prompt):
    """Generate a response using OpenAI."""
    try:
       
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content.strip()
        st.write(f"Response from model: {result}")
        return result
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
        closest_results = query_zilliz(user_vector, top_k=5)

        if closest_results:
            # Step 3: Extract relevant data
            retrieved_texts = extract_relevant_data(closest_results)

            if retrieved_texts:
                # Step 4: Construct prompt with references
                prompt = construct_prompt_with_references(user_query, retrieved_texts)

                # Step 5: Get response from OpenAI
                response = get_response(prompt)
                st.write(f"Response:\n{response}")
            else:
                st.write("No relevant references found.")
        else:
            st.write("No results found in Zilliz.")
    else:
        st.write("Failed to generate embedding.")








