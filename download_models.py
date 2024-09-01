import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain_huggingface import HuggingFaceEmbeddings

def download_models():
    model_name = "t5-large"
    print(f"Downloading {model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    print(f"{model_name} downloaded." )
    
    model_name = "all-mpnet-base-v2"
    print(f"Downloading {model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print(f"{model_name} downloaded." )

if __name__ == "__main__":
    download_models()