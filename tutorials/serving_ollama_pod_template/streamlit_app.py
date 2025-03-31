import os
import streamlit as st
from ollama import Client

client = Client(
    base_url=os.getenv("OLLAMA_ENDPOINT", "http://localhost"),
)


MODEL = "moondream"


def stream_parser(stream):
    for chunk in stream:
        yield chunk["message"]["content"]


query = st.text_input("Enter your question:", value="What is VSCode?")

if query:
    stream = client.chat(
        model=MODEL,
        messages=[{"role": "user", "content": query}],
        stream=True,
    )
    response = st.write_stream(chunk["message"]["content"] for chunk in stream)
