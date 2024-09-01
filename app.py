import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import T5Tokenizer, T5ForConditionalGeneration
from extractor import extract_text_from_pdf_images
import re

# Clean extracted text function
def clean_extracted_text(text):
    cleaned_text = re.sub(r'[_=]+', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n+', ' ', cleaned_text)
    cleaned_text = re.sub(r'TIN: \d+', '', cleaned_text)
    cleaned_text = re.sub(r'Signature: \s+', '', cleaned_text)
    cleaned_text = re.sub(r'Date: \s+', 'Date: ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

# Main function
def main():
    st.set_page_config(page_title="Smart Querying for PDFs")
    st.header("Smart Querying for PDFs")

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for PDF upload
    st.sidebar.header("Upload your PDF file")
    uploaded_pdf = st.sidebar.file_uploader("Drag and drop file here", type="pdf", accept_multiple_files=False)

    if uploaded_pdf and st.sidebar.button("Process PDF"):
        with st.spinner("PDF is processing..."):
            try:
                text = extract_text_from_pdf_images(uploaded_pdf)
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
                return
            
            st.sidebar.write("Text extracted.")
            text_splitter = CharacterTextSplitter(
                separator="\n",  
                chunk_size=500,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            st.sidebar.write("Storing the embeddings in FAISS dataset.")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            if embeddings:
                knowledge_base = FAISS.from_texts(chunks, embeddings)
                st.session_state.knowledge_base = knowledge_base
            st.sidebar.success("PDF is processed.")

    user_question = st.text_input("Ask a question about the PDF: ")
    if st.button("Ask Question") and user_question and "knowledge_base" in st.session_state:
        
        # Display chat history with "Searching..." message
        st.session_state.chat_history.append({"sender": "user", "message": user_question})

        with st.spinner("Searching..."):
            docs = st.session_state.knowledge_base.similarity_search(user_question, k=3)
            context = ""
            for doc in docs:
                context += str(doc)[13:]

            context = clean_extracted_text(context)

            input_text = f"""
                            question: {user_question},
                            context : {context},
                          """

            model_name = "t5-large" 
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
            inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(inputs, max_length=500, num_beams=4, early_stopping=True)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Add bot's answer to chat history
            st.session_state.chat_history.append({"sender": "bot", "message": answer})

    # Display chat history
    if st.session_state.chat_history:
        for entry in st.session_state.chat_history:
            if entry["sender"] == "user":
                st.markdown(f"<div style='text-align: right; margin: 5px;'>{entry['message']} 👨</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: left; margin: 5px;'>🤖 {entry['message']}</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
