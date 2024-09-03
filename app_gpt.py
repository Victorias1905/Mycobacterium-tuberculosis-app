
import openai
import streamlit as st

api_key = st.secrets["general"]["OPENAI_API_KEY"]
# Function to get a response from your fine-tuned model
def get_response(prompt):
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:mtbc-project::A2Vkwrhe",  
    messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip()

# Streamlit app title
st.title("GPT-MTBC gpt-4o-mini")

# Initialize chat history in session state if not already present
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# User input field
user_input = st.text_input("You:", key="user_input")

# If the user provides input, get a response from the model and update the chat history
if user_input:
    response = get_response(user_input)
    st.session_state.chat_history.append({"user": user_input, "bot": response})

# Display the chat history
for chat in st.session_state.chat_history:
    st.write(f"You: {chat['user']}")
    st.write(f"Bot: {chat['bot']}")


