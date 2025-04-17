import os
from langchain_openai import ChatOpenAI
import streamlit as st
 

def get_supervisor_llm(api_key: str = None):
    """
    Fetches the LLM instance dynamically, ensuring it is always up-to-date.
    """
    try:
        if api_key is None:
            api_key = st.session_state.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("⚠️ OpenAI API Key is missing! Please enter it in the UI.")
        llm = ChatOpenAI(model="gpt-4o", api_key=api_key, temperature=0)
        return llm
    except ValueError as e:
        st.error(str(e))
        raise RuntimeError("LLM initialization failed. Check API key.")