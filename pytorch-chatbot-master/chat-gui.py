import streamlit as st

# Initialize an empty list to store chat history
if 'chat_history' not in st.session_state:
  st.session_state['chat_history'] = []

def display_chat_history():
  """Displays the chat history"""
  for message in st.session_state['chat_history']:
    st.chat_message(message["author"], message["content"])

def update_chat_history(user_input, response):
  """Updates the chat history with new messages"""
  st.session_state['chat_history'].append({"author": "user", "content": user_input})
  st.session_state['chat_history'].append({"author": "assistant", "content": response})

st.title("Chat Bot")

# Display chat history
display_chat_history()

# Get user input
user_input = st.chat_input(placeholder="Type your message...")

# Simulate a basic response (Replace with your logic for chatbot functionality)
if user_input:
  response = f"You said: {user_input}"
  update_chat_history(user_input, response)

