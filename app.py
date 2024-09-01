import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import os 
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration,AutoModelForCausalLM,AutoTokenizer,AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from extractor import extract_text_from_pdf
import re

def clean_extracted_text(text):
    cleaned_text = re.sub(r'[_=]+', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text) 
    cleaned_text = cleaned_text.replace('\n', ' ') 

    cleaned_text = re.sub(r'TIN: \d+', '', cleaned_text) 
    cleaned_text = re.sub(r'Signature: \s+', '', cleaned_text) 
    cleaned_text = re.sub(r'Date: \s+', 'Date: ', cleaned_text) 

    cleaned_text = cleaned_text.strip()

    return cleaned_text

def main():
    st.set_page_config(page_title="Smart Querying for PDFs")
    st.header("Smart Querying for PDFs")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    pdf_docs = st.text_input("Enter PDF file path")

    if st.button("Process PDF") and pdf_docs:
        try:
            text = extract_text_from_pdf(pdf_docs)
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return
        
        text_splitter = CharacterTextSplitter(
            separator=" ",  
            chunk_size=500,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        if embeddings:
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            st.session_state.knowledge_base = knowledge_base
        st.write("PDF is processed.")

    user_question = st.text_input("Ask a question about the PDF: ")
    if user_question and "knowledge_base" in st.session_state:
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
        outputs = model.generate(inputs, max_length=200, num_beams=4, early_stopping=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.write(context)
        st.session_state.chat_history.append({"question": user_question, "answer": answer})

    if st.session_state.chat_history:
        st.write("### Chat History")
        for i, entry in enumerate(st.session_state.chat_history, 1):
            st.write(f"👨 {entry['question']}")
            st.write(f"🤖 {entry['answer']}")

if __name__ == '__main__':
    main()