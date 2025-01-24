# Winnie-the-Pooh Chat 🍯🐻

Welcome to a whimsical world where Pooh and friends chitchat right in your browser. This little app uses clever AI embeddings and a vector store to fetch the sweetest snippets from the 1926 classic tales—just the sort of friendly advice Pooh might have up his sleeve!

### See it in action [here](https://winnie-the-pooh-chat.streamlit.app/)! 😀

![🎬 Example Vid](assets/example.webm)

## Features 🌳🐾

- **Streamlit App**: Runs in a tidy web interface. Just type and watch Pooh and friends respond.
- **Vector Search**: Finds relevant hunny-sweet text from the original stories.
- **Anthropic & LangChain**: Handles the chat magic in Pooh’s voice—fluffy and playful.

## Setup 🔧🍯

1. Clone the repo.
2. Install the requirements:

    ```sh
    pip install -r requirements.txt
    ```

3. Set your `ANTHROPIC_API_KEY` environment variable or enter it in the app sidebar.
4. Make sure [winnie.txt] is present in the context folder.
5. Add Streamlit to your PATH (if not already added):

    ```sh
    export PATH=$PATH:~/.local/bin
    ```

6. Run the Streamlit app:

    ```sh
    streamlit run app.py
    ```

## Enjoy 🎉🐝

Let your day be just a bit sunnier as you chat with Pooh and pals!