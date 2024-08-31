import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import os 
from transformers import T5Tokenizer, T5ForConditionalGeneration
from extractor import extract_text_from_pdf_images

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def main():
    model_name = "t5-large" 
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 🤓")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    pdf_docs = st.text_input("Enter PDF file path")

    if st.button("Process PDF") and pdf_docs:
        try:
            text = extract_text_from_pdf_images(pdf_docs)
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
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")
        
        if embeddings:
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            st.session_state.knowledge_base = knowledge_base
        st.write("PDF is processed.")

    user_question = st.text_input("Ask a question about the PDF: ")
    if user_question and "knowledge_base" in st.session_state:
        docs = st.session_state.knowledge_base.similarity_search(user_question, k=1)
        context = str(docs[0])[13:]
        
        input_text = f"Context: {context}, Question: {user_question}"
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

        outputs = model.generate(inputs, max_length=100, num_beams=4, early_stopping=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Store the question and answer in the chat history
        st.session_state.chat_history.append({"question": user_question, "answer": answer})

    # Display the chat history
    if st.session_state.chat_history:
        st.write("### Chat History")
        for i, entry in enumerate(st.session_state.chat_history, 1):
            st.write(f"**Q{i}:** {entry['question']}")
            st.write(f"**A{i}:** {entry['answer']}")

if __name__ == '__main__':
    main()