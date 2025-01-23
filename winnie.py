import streamlit as st
from anthropic import Anthropic
from langchain_anthropic import ChatAnthropic
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

st.set_page_config(page_title="Winnie-the-Pooh Chat", page_icon="🍯")

@st.cache_resource
def load_vectorstore():
    with open('winnie.txt', 'r') as file:
        text = file.read()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

vectorstore = load_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = []

client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def get_relevant_context(query):
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join(doc.page_content for doc in docs)

SYSTEM_TEMPLATE = """You are a friendly chatbot who speaks in the style of characters from the original 1926 
Winnie-the-Pooh book by A.A. Milne. Base your responses on the following relevant context from the original book:

{context}

Only reference events and characters from this original book. Maintain a warm, friendly tone while avoiding any copyrighted content from later works."""

st.title("Chat with Pooh and Friends 🍯")
st.caption("Based on A.A. Milne's original 1926 public domain work")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to discuss?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    context = get_relevant_context(prompt)
    system_prompt = SYSTEM_TEMPLATE.format(context=context)
    messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        stream = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[
                {"role": m["role"], "content": m["content"]} 
                for m in messages
            ],
            stream=True
        )

        for chunk in stream:
            if chunk.delta.text:
                full_response += chunk.delta.text
                message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

with st.sidebar:
    st.header("About")
    st.markdown("""
    This chat app uses content only from the original 1926 Winnie-the-Pooh book, 
    which entered the public domain in 2022.""")