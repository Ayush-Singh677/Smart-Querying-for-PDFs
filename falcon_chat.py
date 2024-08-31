import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os 
from extractor import extract_text_from_pdf_images

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def main():
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 🤓")

    pdf_docs = st.text_input("Enter PDF file path")

    if st.button("Process PDF") and pdf_docs:
        try:
            text = extract_text_from_pdf_images(pdf_docs)

            with open("ANS.txt", "w") as file:
                file.write(text)

        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return
        
        text_splitter = CharacterTextSplitter(
            separator="\n",  
            chunk_size=1000,
            chunk_overlap=300,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        embeddings = HuggingFaceEmbeddings()
        llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.1, "max_length": 1024})
        
        if embeddings:
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            st.session_state.knowledge_base = knowledge_base
            st.session_state.llm = llm
        st.write("PDF is processed.")

    user_question = st.text_input("Ask a question about the PDF: ")
    if user_question and "knowledge_base" in st.session_state:
        docs = st.session_state.knowledge_base.similarity_search(user_question,k=1)
        
        # Define a streamlined prompt template
        prompt = f"""{user_question} + 
                     ***Answer strictly from the context don't hallucinate!!!!!!***
                     """

        chain = load_qa_chain(st.session_state.llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=prompt)
        # response = chain.run(question=formatted_prompt)

        # Clean up response if necessary
        st.write(response)

if __name__ == '__main__':
    main()