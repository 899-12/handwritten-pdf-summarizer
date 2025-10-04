import streamlit as st
import fitz  # PyMuPDF
import easyocr
from transformers import pipeline
import numpy as np
from PIL import Image

# Initialize OCR and summarizer
reader = easyocr.Reader(['en'], gpu=False)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Title
st.title("📝 Handwritten PDF Digitizer & Summarizer")

# Upload PDF
uploaded_file = st.file_uploader("Upload a Handwritten PDF", type=["pdf"])

if uploaded_file:
    st.info("Extracting text from PDF...")

    try:
        # Open PDF with PyMuPDF
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")

        full_text = ""

        for i, page in enumerate(doc):
            # Convert page to image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Show page image
            st.image(img, caption=f"Page {i+1}", use_column_width=True)

            # OCR extraction
            result = reader.readtext(np.array(img), detail=0)
            extracted_text = ' '.join(result)
            st.text_area(f"🖹 Text from Page {i+1}", extracted_text, height=200)
            full_text += extracted_text + "\n"

        # Download extracted text
        st.download_button("⬇️ Download Extracted Text (.txt)", full_text, file_name="extracted_text.txt")

        # Summarization
        if st.button("📄 Generate Summary"):
            if len(full_text.strip()) > 0:
                st.info("Generating summary using AI...")

                custom_prompt = f"Summarize this medical prescription:\n{full_text}"

                summary = summarizer(custom_prompt, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
                st.success("Summary Generated:")
                st.write(summary)
                st.download_button("⬇️ Download Summary", summary, file_name="summary.txt")
            else:
                st.warning("No text found to summarize.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
