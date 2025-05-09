import os
import streamlit as st
from typing import Dict, List, Tuple, Any, Optional
from pydantic import BaseModel

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Set page configuration
st.set_page_config(page_title="LangGraph Chat", page_icon="💬")

# Initialize session state for chat history and model selection if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful AI assistant.")
    ]
    
if "model_name" not in st.session_state:
    st.session_state.model_name = "gpt-3.5-turbo"
    
if "custom_model" not in st.session_state:
    st.session_state.custom_model = ""

# Define the state schema for our graph
class GraphState(BaseModel):
    """Represents the state of our conversation graph."""
    messages: List[Any]
    
# Define the nodes for our conversation graph
def get_llm_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Get a response from the LLM based on the conversation history."""
    # Initialize the ChatOpenAI model with the selected model
    model_name = st.session_state.model_name
    
    # If using custom model, use the custom model name instead
    if model_name == "custom" and st.session_state.custom_model:
        model_name = st.session_state.custom_model
        
        # Check if we need to use a custom endpoint for the model
        endpoint_pattern = os.environ.get("INTERNAL_APP_ENDPOINT_PATTERN")
        if endpoint_pattern:
            # Construct the URL using the pattern and the custom model name as app_fqdn
            base_url = endpoint_pattern.format(app_fqdn=model_name)
            llm = ChatOpenAI(model=model_name, temperature=0.7, base_url=base_url)
        else:
            llm = ChatOpenAI(model=model_name, temperature=0.7)
    else:
        llm = ChatOpenAI(model=model_name, temperature=0.7)
    
    # Get response from LLM
    response = llm.invoke(state.messages)
    
    # Add the AI's response to the messages
    return {"messages": state.messages + [response]}

# Build the graph
def build_graph():
    """Build the conversation graph."""
    # Initialize the graph
    graph = StateGraph(GraphState)
    
    # Add the LLM node
    graph.add_node("llm", get_llm_response)
    
    # Set the entry point
    graph.set_entry_point("llm")
    
    # Add an edge from the LLM back to itself (for continuous conversation)
    graph.add_edge("llm", END)
    
    # Compile the graph
    return graph.compile()

# Create the graph
graph = build_graph()

# Streamlit UI
st.title("💬 LangGraph Chat")
st.subheader("Chat with an AI assistant powered by LangGraph and OpenAI")

# Sidebar for model selection
with st.sidebar:
    st.title("Model Settings")
    model_option = st.radio(
        "Select Model:",
        options=["gpt-3.5-turbo", "custom"],
        index=0 if st.session_state.model_name == "gpt-3.5-turbo" else 1,
        key="model_selection"
    )
    
    # If custom model is selected, show text input for custom model name
    if model_option == "custom":
        custom_model = st.text_input(
            "Enter union-deployed app name:",
            value=st.session_state.custom_model,
            key="custom_model_input"
        )
        if custom_model:
            st.session_state.custom_model = custom_model
    
    # Update the model name in session state
    st.session_state.model_name = model_option
    
    # Display current model
    current_model = st.session_state.custom_model if model_option == "custom" and st.session_state.custom_model else model_option
    st.write(f"Current model: **{current_model}**")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue  # Skip system messages in the UI
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

# Accept user input
if prompt := st.chat_input("What would you like to talk about?"):
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Run the graph with the current state
        result = graph.invoke({"messages": st.session_state.messages})
        
        # Get the last message (the AI's response)
        ai_message = result["messages"][-1]
        message_placeholder.markdown(ai_message.content)
        
        # Add AI response to chat history
        st.session_state.messages.append(ai_message)