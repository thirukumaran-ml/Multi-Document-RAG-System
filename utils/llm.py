import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def load_llm():

    # Try Streamlit Secrets first (deployment)
    try:
        import streamlit as st

        api_key = st.secrets.get(
            "GOOGLE_API_KEY",
            os.getenv("GOOGLE_API_KEY")
        )

    except Exception:

        api_key = os.getenv(
            "GOOGLE_API_KEY"
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0
    )

    return llm