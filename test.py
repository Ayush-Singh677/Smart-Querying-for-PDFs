from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from transformers import pipeline
import streamlit as st

# Function to extract text from PDF
def extract_images_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    return images

def ocr_image_tesseract(image):
    text = pytesseract.image_to_string(image)
    return text

def extract_text_from_pdf_images(pdf_path):
    images = extract_images_from_pdf(pdf_path)
    all_text = ""
    for img in images:
        text = ocr_image_tesseract(img)
        all_text += text + "\n"
    return all_text

# Load a pre-trained question-answering model
qa_pipeline = pipeline("question-answering",model = "deepset/roberta-base-squad2")

def answer_question(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

# Streamlit app
st.title("PDF Answering Chatbot")

# Extract text from PDF
pdf_path = st.text_input("Path to your PDF file:")
if pdf_path:
    text = extract_text_from_pdf_images(pdf_path)
    st.write("PDF text extracted successfully!")

    # User input
    user_query = st.text_input("Ask a question:")
    if user_query:
        answer = answer_question(user_query, text)
        st.write(answer)