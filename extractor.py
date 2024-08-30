from pdf2image import convert_from_path
import pytesseract
from PIL import Image

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
