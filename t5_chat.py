import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from extractor import extract_text_from_pdf_images

# Set Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def main():
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 🤓")

    pdf_path = st.text_input("Enter PDF file path")

    if st.button("Process PDF") and pdf_path:
        try:
            text = extract_text_from_pdf_images(pdf_path)

            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=300,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            st.write(f"Number of chunks: {len(chunks)}")

            # Create embeddings and FAISS index
            embeddings = HuggingFaceEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            st.session_state.knowledge_base = knowledge_base
            st.write("PDF is processed.")

        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return
        
    query = st.text_input("Enter your query:")

    if st.button("Ask Question") and query:
        if 'knowledge_base' not in st.session_state:
            st.error("No knowledge base found. Please process a PDF first.")
            return

        # Initialize QA model from Hugging Face Hub
        model_name = "distilbert-base-uncased-distilled-squad"  # Ensure this model exists
        llm = HuggingFaceHub(repo_id="google-t5/t5-large",model_kwargs={"temperature": 0.1, "max_length": 1024})

        # Initialize the RetrievalQA chain
        retriever = st.session_state.knowledge_base.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )

        # Get the answer
        answer = qa_chain.run(query=query)

        st.write(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()