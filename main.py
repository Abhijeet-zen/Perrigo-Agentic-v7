"""
main.py

Entry point for the Multi-Agent AI system.
Prompts the user for API key before launching UI and agents.
"""

import os
import openai
import logging
import traceback
import streamlit as st
from config import setup_logging
from src.ui import run_ui
from PIL import Image
from src.utils.openai_api import get_supervisor_llm

setup_logging()

def check_openai_api_key(api_key):
    """Validate OpenAI API key."""
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
        os.environ["OPENAI_API_KEY"] = api_key
        return True
    except openai.AuthenticationError as e:
        # st.error(f"Error occurred: {str(e)}")
        return False
    
    

def main():
    # Initialize session state
    if "sidebar_collapsed" not in st.session_state:
        st.session_state.sidebar_collapsed = False

    # Set Streamlit layout
    st.set_page_config(
        page_title="GenAI Answer Bot",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed" if st.session_state.sidebar_collapsed else "expanded"
    )

    # Styling
    st.markdown("""
        <style>
            section.main > div:first-child {
                padding-top: 3rem;
                max-width: 100rem !important;
            }
            [data-testid="stSidebar"] {
                width: 300px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar: Logo
    try:
        logo = Image.open("Images/perrigo-logo.png")
        st.sidebar.image(logo, width=80)
    except Exception:
        st.sidebar.error("Logo image not found.")

    # Sidebar: API Key Input
    st.sidebar.header("⚙️ Configuration")
    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password", key="api_key_input")

    if not api_key:
        st.info("Please enter your OpenAI API key in the sidebar to continue.")
        return

    # Validate key
    if not check_openai_api_key(api_key):
        st.error("❌ Invalid API Key. Please check and try again.")
        return

    # Save in session and env
    st.session_state["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key

    # Collapse sidebar
    if not st.session_state.sidebar_collapsed:
        st.session_state.sidebar_collapsed = True
        st.rerun()

    # Get LLM instance
    try:
        st.session_state["llm"] = get_supervisor_llm(api_key)
    except ValueError as e:
        st.error(f"❌ Invalid API Key: {e}")
        return

    logging.info("✅ OpenAI API Key set. Starting the Multi-Agent AI System...")

    # Run app UI
    try:
        run_ui()
    except Exception as e:

        full_error = traceback.format_exc()
        logging.error(f"Error starting the UI:\n{full_error}")
        st.error(f"An error occurred while launching the application:\n{full_error}")

        # logging.error(f"Error starting the UI: {e}")
        # st.error(f"An error occurred while launching the application: {e}")


if __name__ == "__main__":
    main()
