import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from huggingface_hub import login
from extractor import extract_text_from_pdf_images
from sentence_transformers import SentenceTransformer
# Bug: API key should not be hardcoded directly in the code. Use environment variables or Streamlit's secrets management instead.
login("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")

def get_text_chunks(text):
    # Ensure the text is split into manageable chunks to avoid hitting model limits.
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=400,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # Embedding selection based on your preference for better model performance.
    # embeddings = OpenAIEmbeddings()  # Commented out: Only needed if you decide to use OpenAI's embeddings.
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    
    # FAISS vector store creation for efficient similarity search.
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # Switch from OpenAI to HuggingFace model for LLM.
    # llm = ChatOpenAI()  # Commented out: Used if you want to switch to OpenAI's model.
    llm = HuggingFaceHub(repo_id="google/flan-t5-base",huggingfacehub_api_token="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", model_kwargs={"temperature": 0.5, "max_length": 512})

    # Memory object to keep track of conversation history.
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    # Create a conversation chain that retrieves relevant text from the vector store.
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    # Handle user input and update the chat history accordingly.
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display the chat history in an alternating user-bot format.
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    # Set up the Streamlit page with appropriate title and icon.
    # load_dotenv()  # Commented out: Uncomment if you need to load environment variables from a .env file.
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables if not already initialized.
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.text_input("Enter pdf file path")
        
        if st.button("Process"):
            with st.spinner("Processing"):
                # Extract text from the PDF images using a custom function.
                raw_text = extract_text_from_pdf_images(pdf_docs)
                st.write("Text Extracted.")
                # Split the extracted text into manageable chunks.
                text_chunks = get_text_chunks(raw_text)
                st.write("Text broken into chunks.")
                # Create a vector store for the chunks.
                vectorstore = get_vectorstore(text_chunks)
                st.write("Embeddings Stored.")
                # Set up the conversational retrieval chain.
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

if __name__ == '__main__':
    main()