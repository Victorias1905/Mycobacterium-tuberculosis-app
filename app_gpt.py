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
connections.connect("default", uri=zilliz_uri, token=zilliz_token) 
zilliz_client = MilvusClient(uri=zilliz_uri, token=zilliz_token)
zilliz_client.describe_collection(collection_name=collection_name)
collection = Collection(collection_name)  

def get_response(prompt, model):
    """Generate a response using OpenAI."""
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def get_embedding(text):
    """Generate embedding for text using OpenAI."""
    embedding_response = openai.OpenAI(api_key=api_key).embeddings.create(
        input=text, 
        model="text-embedding-3-small")
    vector = embedding_response.data[0].embedding
    return vector 

def query_zilliz(query_embedding, top_k=5):
    results = collection.search(
        data=[query_embedding],
        anns_field="vector",  # Correct field name
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},  # Use COSINE metric
        limit=top_k
    )
    return results

def construct_prompt_with_references(query, references):
    """Constructs a prompt with user query and retrieved references."""
    formatted_references = "\n".join(references)
    return (
        f"User query: {query}\n\n"
        f"Relevant information from the database:\n{formatted_references}\n\n"
        "Please generate a response based on the above information."
    )

# Streamlit App
st.title("Compare OpenAI Models with and without Zilliz")

col1, col2 = st.columns(2)

# Model 1 - Without references
with col1:
    st.markdown("### Model 1: Without references")

    if 'chat_history_model1' not in st.session_state:
        st.session_state.chat_history_model1 = []

    user_input_model1 = st.text_input("You (Model 1):", key="user_input_model1")

    if user_input_model1:
        response_model1 = get_response(user_input_model1, "ft:gpt-4o-mini-2024-07-18:mtbc-project::AkyZCr4h")
        st.session_state.chat_history_model1.append({"user": user_input_model1, "bot": response_model1})

    for chat in st.session_state.chat_history_model1:
        st.write(f"You: {chat['user']}")
        st.write(f"Bot: {chat['bot']}")

# Model 2 - With references
with col2:
    st.markdown("### Model 2: With references")

    if 'chat_history_model2' not in st.session_state:
        st.session_state.chat_history_model2 = []

    user_input_model2 = st.text_input("You (Model 2):", key="user_input_model2")

    if user_input_model2:
        # Generate embedding
        query_embedding = get_embedding(user_input_model2)
        zilliz_results = query_zilliz(query_embedding, top_k=5)
        
        retrieved_texts = []
        for hits in zilliz_results:
            for hit in hits:
                retrieved_texts.append(collection.get(hit.id)[0][embedding_field])
        # Construct prompt with references
        prompt_with_references = construct_prompt_with_references(user_input_model2, retrieved_texts)

        # Get response from the fine-tuned model
        response_model2 = get_response(prompt_with_references, "ft:gpt-4o-mini-2024-07-18:mtbc-project::Akwtgx7I")
        st.session_state.chat_history_model2.append({"user": user_input_model2, "bot": response_model2})

    for chat in st.session_state.chat_history_model2:
        st.write(f"You: {chat['user']}")
        st.write(f"Bot: {chat['bot']}")



