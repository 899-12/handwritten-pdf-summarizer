import streamlit as st
from pdf2image import convert_from_bytes
import easyocr
from transformers import pipeline
import numpy as np
from PIL import Image
import io

# Initialize OCR and summarizer
reader = easyocr.Reader(['en'])
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Title
st.title("📝 Handwritten PDF Digitizer & Summarizer")

# Upload PDF
uploaded_file = st.file_uploader("Upload a Handwritten PDF", type=["pdf"])

if uploaded_file:
    st.info("Extracting text from PDF...")
    
    # Convert PDF to images
    images = convert_from_bytes(uploaded_file.read())
    
    full_text = ""
    
    for i, img in enumerate(images):
        st.image(img, caption=f"Page {i+1}", use_column_width=True)
        result = reader.readtext(np.array(img), detail=0)
        extracted_text = ' '.join(result)
        st.text_area(f"🖹 Text from Page {i+1}", extracted_text, height=200)
        full_text += extracted_text + "\n"

    st.download_button("⬇️ Download Extracted Text (.txt)", full_text, file_name="extracted_text.txt")

    if st.button("📄 Generate Summary"):
        if len(full_text.strip()) > 0:
            st.info("Generating summary using AI...")
            
            # Prompt Engineering for summarization
            custom_prompt = f"Summarize the content and what is the presciption for:\n{full_text}"
            
            # Generate summary with the custom prompt
            summary = summarizer(custom_prompt, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            
            st.success("Summary Generated:")
            st.write(summary)
            st.download_button("⬇️ Download Summary", summary, file_name="summary.txt")
        else:
            st.warning("No text found to summarize.")
