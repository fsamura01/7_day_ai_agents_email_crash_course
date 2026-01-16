import streamlit as st
import asyncio
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ingest import run_ingestion
from search_tools import initialize_search_indexes
from search_agent import create_agent
from logs import log_interaction

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Repo Docs Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Repository Documentation Agent")
st.markdown("Ask questions about your GitHub repository's documentation.")

# --- Environment Setup (Streamlit Secrets) ---
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass # Locally we use .env, so it's fine if secrets are missing

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

if "indexed" not in st.session_state:
    # Check if files already exist on disk
    if os.path.exists("chunks.json"):
        st.session_state.indexed = initialize_search_indexes()
    else:
        st.session_state.indexed = False

if "agent" not in st.session_state:
    st.session_state.agent = create_agent()

# --- Sidebar: Configuration & Ingestion ---
with st.sidebar:
    st.header("⚙️ Configuration")
    repo_owner = st.text_input("GitHub Owner", value="fsamura01")
    repo_name = st.text_input("Repository Name", value="task-manager-app")
    
    if st.button("🚀 Ingest Repository"):
        with st.status("Ingesting documentation...", expanded=True) as status:
            st.write("Fetching and chunking data...")
            run_ingestion(repo_owner, repo_name)
            st.write("Initializing search indexes...")
            success = initialize_search_indexes()
            st.session_state.indexed = success
            if success:
                status.update(label="Ingestion Complete!", state="complete", expanded=False)
                st.success("Repository indexed successfully!")
            else:
                status.update(label="Ingestion Failed", state="error")
                st.error("Failed to initialize search indexes.")

    st.divider()
    if st.session_state.indexed:
        st.success("✅ Repository Indexed")
    else:
        st.warning("⚠️ No data indexed. Please run ingestion.")

# --- Main Chat Interface ---
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know about the repo?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if not st.session_state.indexed:
        with st.chat_message("assistant"):
            st.error("I don't have any documentation to search. Please ingest a repository in the sidebar.")
    else:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            
            async def run_and_stream():
                accumulated_text = ""
                # Keep only the last 6 messages (3 turns) to save tokens
                history_window = st.session_state.agent_messages[-6:] if st.session_state.agent_messages else []
                
                async with st.session_state.agent.run_stream(
                    prompt, 
                    message_history=history_window
                ) as result:
                    async for message in result.stream_output():
                        accumulated_text = message
                        placeholder.markdown(accumulated_text)
                    return result, accumulated_text

            try:
                # Execute the streaming logic
                result, response_text = asyncio.run(run_and_stream())
                
                # Add assistant response to UI history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Update agent history for the next turn
                st.session_state.agent_messages = result.all_messages()
                
                # Log the interaction
                log_interaction(st.session_state.agent, result.new_messages())
            except Exception as e:
                st.error(f"Error: {e}")

# --- Footer ---
st.divider()
st.caption("Day 6: Refactoring & Publication - Powered by Pydantic AI & Streamlit")
