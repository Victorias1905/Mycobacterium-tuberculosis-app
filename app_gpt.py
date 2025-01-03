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
embedding_field = "Mycobacterium"  # Field name for embeddings in Zilliz

# Connect to Zilliz Cloud
try:
    connections.connect("default", uri=zilliz_uri, token=zilliz_token)
    zilliz_client = MilvusClient(uri=zilliz_uri, token=zilliz_token)
    collection = Collection(collection_name)
    st.write("Connected to Zilliz Cloud successfully!")
except Exception as e:
    st.write(f"Failed to connect to Zilliz Cloud: {e}")

def get_response(prompt, model):
    """Generate a response using OpenAI."""
    try:
        st.write(f"Prompt sent to model: {prompt}")
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

def get_embedding(text):
    """Generate embedding for text using OpenAI."""
    try:
        st.write(f"Generating embedding for: {text}")
        embedding_response = openai.OpenAI(api_key=api_key).embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        vector = embedding_response.data[0].embedding
        st.write(f"Generated embedding: {vector[:5]}... (truncated for display)")
        return vector
    except Exception as e:
        st.write(f"Error generating embedding: {e}")
        return None

def query_zilliz(query_embedding, top_k=5):
    """Query Zilliz database."""
    try:
        st.write(f"Querying Zilliz with embedding: {query_embedding[:5]}... (truncated for display)")
        results = collection.search(
            data=[query_embedding],
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k
        )
        st.write(f"Search results: {results}")
        return results
    except Exception as e:
        st.write(f"Error querying Zilliz: {e}")
        return None

def construct_prompt_with_references(query, references):
    """Constructs a prompt with user query and retrieved references."""
    formatted_references = "\n".join(references)
    prompt = (
        f"User query: {query}\n\n"
        f"Relevant information from the database:\n{formatted_references}\n\n"
        "Please generate a response based on the above information."
    )
    st.write(f"Constructed prompt: {prompt}")
    return prompt

# Streamlit App
st.title("Debugging Zilliz Retrieval")

user_input = st.text_input("Enter your query:")

if user_input:
    # Generate embedding
    query_embedding = get_embedding(user_input)

    if query_embedding:
        # Query Zilliz
        zilliz_results = query_zilliz(query_embedding, top_k=5)

        if zilliz_results:
            retrieved_texts = []
            for hits in zilliz_results:
                for hit in hits:
                    try:
                        result = collection.query(expr=f"Auto_id == {hit.id}",output_fields=["vector"])
                        if result:
                            st.write(f"Retrieved document for ID {hit.id}: {result}")
                            retrieved_texts.append(result[0][embedding_field])
                        else:
                            st.write(f"No document found for ID {hit.id}")
                    except Exception as e:
                        st.write(f"Error retrieving document for ID {hit.id}: {e}")

            # Construct the prompt with references
            if retrieved_texts:
                prompt_with_references = construct_prompt_with_references(user_input, retrieved_texts)
                response_with_references = get_response(prompt_with_references, "ft:gpt-4o-mini-2024-07-18:mtbc-project::Akwtgx7I")
                st.write(f"Response with references: {response_with_references}")
            else:
                st.write("No relevant references retrieved.")



