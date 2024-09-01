import pytesseract
from PIL import Image
import cv2
from pdf2image import convert_from_bytes
import numpy as np

def convert_pdf_to_images(pdf_file):
    images = convert_from_bytes(pdf_file.read())
    return images

def preprocess_image(image):
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return binary_image

def extract_text_from_image(image):
    preprocessed_image = preprocess_image(image)
    pil_image = Image.fromarray(preprocessed_image)
    text = pytesseract.image_to_string(pil_image)
    return text

def extract_text_from_pdf(pdf_file):
    images = convert_pdf_to_images(pdf_file)
    full_text = ""
    for image in images:
        text = extract_text_from_image(image)
        full_text += text + "\n"
    return full_text