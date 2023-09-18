import streamlit as st
from ocr_processor import OCRProcessor
from ui import display_input_form

def main():
    st.set_page_config(
        page_title="OCR App",
        page_icon=":memo:",  # Add an icon
        layout="wide"  # Use a wider layout
    )
    st.title("OCR App")
    st.markdown("""
    Optical Character Recognition (OCR) allows you to extract text from images and PDFs. 
    Upload an image or PDF to get started!
    """)

    ocr_processor = OCRProcessor()
    input_type, file_path = display_input_form()

    if file_path:
        images, text_regions, extracted_text = ocr_processor.process_file(input_type, file_path)

        if images:
            display_results(images, text_regions, extracted_text)

if __name__ == "__main__":
    main()

