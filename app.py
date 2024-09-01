import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import T5Tokenizer, T5ForConditionalGeneration
from extractor import extract_text_from_pdf
import re

def clean_extracted_text(text):
    cleaned_text = re.sub(r'[_=]+', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'\n+', ' ', cleaned_text)
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

    st.sidebar.header("Upload your PDF file")
    uploaded_pdf = st.sidebar.file_uploader("Drag and drop file here", type="pdf", accept_multiple_files=False)
    
    responses = []
    if uploaded_pdf and st.sidebar.button("Process PDF"):
        with st.spinner("PDF is processing..."):
            try:
                text = extract_text_from_pdf(uploaded_pdf)
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
        
        responses = []
        responses.append({"sender": "user", "message": user_question})

        with st.spinner("Searching..."):
            docs_with_scores = st.session_state.knowledge_base.similarity_search_with_score(user_question, k=3)
            context = ""

            threshold = 1.5  
            found_match = False

            for doc, score in docs_with_scores:
                if score <= threshold:
                    context += str(doc)[13:]
                    found_match = True

            if not found_match:
                answer = "No match found for your query in the document. Please be more specific."
                st.session_state.chat_history.append({"sender": "bot", "message": answer})

            else:
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

                responses.append({"sender": "bot", "message": answer})
    
    responses = reversed(responses)
    for response in responses:
        st.session_state.chat_history.append(response)
    if st.session_state.chat_history:
        for entry in reversed(st.session_state.chat_history):
            if entry["sender"] == "user":
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: flex-end; margin: 5px;'>
                        <div style='background-color: #007AFF; border-radius: 5px; padding: 10px; max-width: max-content; color: white;'>
                            {entry['message']} 👨
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style='display: flex; justify-content: flex-start; margin: 5px;'>
                        <div style='background-color: #3A3A3A; border-radius: 5px; padding: 10px; max-width: max-content; color: white;'>
                            🤖 {entry['message']}
                        </div>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )


if __name__ == '__main__':
    main()
