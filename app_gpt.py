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
        model="text-embedding-3-small"
    )
    vector = embedding_response.data[0].embedding
    return vector

def query_zilliz(query_embedding, top_k=5):
    """Search for similar documents in the Zilliz vector database."""
    results = collection.search(
        data=[query_embedding],
        anns_field="vector",  # Ensure this matches your actual vector field name
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
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
st.title("Chatbot with Zilliz References")

# Maintain chat history in session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Text input for user's query
user_input = st.text_input("Enter your question:")

if user_input:
    # Generate embedding for the user query
    query_embedding = get_embedding(user_input)
    zilliz_results = query_zilliz(query_embedding, top_k=5)

    # Retrieve texts from Zilliz
    retrieved_texts = []
    for hits in zilliz_results:
        for hit in hits:
            # Adjust the field/fields you want to retrieve here
            result = collection.query(
                expr=f"id == {hit.id}",
                output_fields=[embedding_field]
            )
            if result:
                # Extract the actual text content from result
                # Adjust indexing/field name as needed for your schema
                retrieved_texts.append(result[0].get(embedding_field, ""))

    # Construct the prompt with references
    prompt_with_references = construct_prompt_with_references(user_input, retrieved_texts)
    
    # Display the combined prompt for reference
    st.write("**Prompt with references:**")
    st.write(prompt_with_references)

    # Get the response from the language model (Model 2)
    response = get_response(prompt_with_references, "ft:gpt-4o-mini-2024-07-18:mtbc-project::Akwtgx7I")

    # Update chat history
    st.session_state.chat_history.append({"user": user_input, "bot": response})

# Display the chat history
for chat in st.session_state.chat_history:
    st.write(f"**You:** {chat['user']}")
    st.write(f"**Bot:** {chat['bot']}")



