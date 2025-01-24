"""
Streamlit application for chatting with Winnie-the-Pooh characters using the Anthropic API.
The app uses FAISS for vector similarity search and LangChain for text processing.
"""

import streamlit as st
import logging
from anthropic import Anthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(
    level=logging.INFO,
    filename='app.log',
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_session_state():
    """Initialize Streamlit session state variables with default values."""
    defaults = {
        "messages": [],
        "api_key": None,
        "vectorstore": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def load_vectorstore():
    """
    Load and cache the vector store for similarity search.
    
    Returns:
        FAISS: Loaded vector store with embedded text chunks
    """
    if st.session_state.vectorstore is None:
        with open('winnie.txt', 'r') as file:
            text = file.read()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        ).split_text(text)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
    return st.session_state.vectorstore

def get_relevant_context(query, vectorstore):
    """
    Retrieve relevant context from the vector store based on query similarity.
    
    Args:
        query (str): User's input query
        vectorstore (FAISS): Vector store containing embedded text chunks
    
    Returns:
        str: Concatenated relevant text chunks
    """
    docs = vectorstore.similarity_search(query, k=2)
    return "\n\n".join(doc.page_content for doc in docs)

def process_message(prompt, client, vectorstore):
    """
    Process user message and generate AI response using Anthropic's API.
    
    Args:
        prompt (str): User's input message
        client (Anthropic): Initialized Anthropic client
        vectorstore (FAISS): Vector store for context retrieval
    
    Returns:
        str or None: Generated response or None if error occurs
    """
    context = get_relevant_context(prompt, vectorstore)
    system_prompt = f"""You are one of the characters from this context.:

    {context}

    There is a narrator who describes your response. Only reference events and characters from this context. Maintain the tone and attitude of the character you are while avoiding any copyrighted content from later works."""
    
    logger.info(f"Context: {context[:200]}...")
    logger.info(f"System prompt: {system_prompt[:200]}...")
    
    messages = [{"role": "user", "content": prompt}]
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
                stream=True
            )

            for chunk in stream:
                if chunk.type == "content_block_start" or chunk.type == "content_block_delta":
                    if hasattr(chunk, "delta") and chunk.delta.text:
                        full_response += chunk.delta.text
                        message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            logger.info(f"Response length: {len(full_response)}")
            return full_response
            
        except Exception as e:
            logger.error(f"Error in message processing: {str(e)}")
            st.error("An error occurred while processing your message. Please try again.")
            return None

def main():
    """
    Main application function that sets up the Streamlit interface and handles the chat flow.
    """
    st.set_page_config(page_title="Winnie-the-Pooh Chat", page_icon="🍯")
    init_session_state()

    st.title("Chat with Pooh and Friends 🍯")
    st.caption("Based on A.A. Milne's original 1926 public domain work")

    api_key = st.text_input("Enter Anthropic API Key", type="password")
    if api_key:
        st.session_state.api_key = api_key

    if not st.session_state.api_key:
        st.warning("Please enter your Anthropic API Key to continue.")
        return

    vectorstore = load_vectorstore()
    client = Anthropic(api_key=st.session_state.api_key)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to discuss?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = process_message(prompt, client, vectorstore)
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})

    with st.sidebar:
        st.header("About")
        st.markdown("Chat with the original 1926 Winnie-the-Pooh book characters.")

if __name__ == "__main__":
    main()