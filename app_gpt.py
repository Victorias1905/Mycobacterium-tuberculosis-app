import openai
import streamlit as st

api_key = st.secrets["general"]["OPENAI_API_KEY"]


def get_response(prompt, model):
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,  
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Model 1: Without references")
    
    if 'chat_history_model1' not in st.session_state:
        st.session_state.chat_history_model1 = []

    user_input_model1 = st.text_input("You (Model 1):", key="user_input_model1")

    if user_input_model1:
        response_model1 = get_response(user_input_model1, "ft:gpt-4o-mini-2024-07-18:mtbc-project::A2Vkwrhe")
        st.session_state.chat_history_model1.append({"user": user_input_model1, "bot": response_model1})

    for chat in st.session_state.chat_history_model1:
        st.write(f"You: {chat['user']}")
        st.write(f"Bot: {chat['bot']}")

# Second Column - Model 2
with col2:
    st.markdown("### Model 2: With references") 
    
    if 'chat_history_model2' not in st.session_state:
        st.session_state.chat_history_model2 = []

    user_input_model2 = st.text_input("You (Model 2):", key="user_input_model2")

    if user_input_model2:
        response_model2 = get_response(user_input_model2, "ft:gpt-4o-mini-2024-07-18:mtbc-project::AQhs3HpR")  
        st.session_state.chat_history_model2.append({"user": user_input_model2, "bot": response_model2})

    for chat in st.session_state.chat_history_model2:
        st.write(f"You: {chat['user']}")
        st.write(f"Bot: {chat['bot']}")


