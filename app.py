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
        with open('context/winnie.txt', 'r') as file:
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
    system_prompt = f'''Speaking with the user you are one of the characters from this context:

    {context}

    Respond in the first person as the character you are embodying.
    That response should be in style and tone true to the orginal context.
    Only reference events and characters from the context.
    You are not aware of the users previous prompts.
    You are always speaking to the user.
    Maintain the tone and attitude of the character you are while avoiding any copyrighted content from later works.'''
    
    logger.info(f"Context: {context[:200]}...")
    logger.info(f"System prompt: {system_prompt[:200]}...")
    
    messages = [{"role": "user", "content": prompt}]
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            stream = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=2048,
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

    # Sidebar elements
    with st.sidebar:
        st.markdown("### Settings")
        api_key = st.text_input("Anthropic API Key", type="password", key="api_key_input")
        if api_key:
            st.session_state.api_key = api_key
        st.markdown("### About")
        st.markdown("Chat with the original 1926 Winnie-the-Pooh book characters using Anthropic's Haiku AI.")
    


    # Main content
    st.image("header.jpg")
    st.title("A chat with Winnie the Pooh and Friends 🍯")
    st.caption("Based on A.A. Milne's original 1926 public domain work")
    
    prompt = st.chat_input("What would you like to discuss?", key="chat_input")
    
    if prompt and not st.session_state.api_key:
        st.error("Please enter your Anthropic API Key to continue")
        return

    vectorstore = load_vectorstore()
    
    if st.session_state.api_key:
        client = Anthropic(api_key=st.session_state.api_key)
        
    # Process new messages
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        response = process_message(prompt, client, vectorstore)
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            # with st.chat_message("assistant"):
            #     st.markdown(response)

if __name__ == "__main__":
    main()