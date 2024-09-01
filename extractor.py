import pytesseract
from PIL import Image
import cv2
from pdf2image import convert_from_path
import os

# Convert PDF to images
def convert_pdf_to_images(pdf_path):
    images = convert_from_path(pdf_path)
    image_paths = []
    for i, image in enumerate(images):
        image_path = f"page_{i+1}.png"
        image.save(image_path, 'PNG')
        image_paths.append(image_path)
    return image_paths

# Preprocess the image for better OCR accuracy
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, binary_image = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return binary_image

# Apply OCR on the image
def extract_text_from_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    text = pytesseract.image_to_string(preprocessed_image)
    return text

# Process the PDF and extract text from each page
def extract_text_from_pdf(pdf_path):
    image_paths = convert_pdf_to_images(pdf_path)
    full_text = ""
    for image_path in image_paths:
        text = extract_text_from_image(image_path)
        full_text += text + "\n"
        os.remove(image_path)  # Clean up the image files after processing
    return full_text