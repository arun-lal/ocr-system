import streamlit as st
import tempfile

def display_input_form():
    input_type = st.sidebar.radio("Select input type:", ("Upload Image/PDF", "Enter File Path"))
    file_path = None  # Initialize file_path variable

    if input_type == "Upload Image/PDF":
        st.sidebar.markdown("### Upload Image/PDF")
        uploaded_file = st.sidebar.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])
        
        if uploaded_file is not None:
            # Determine the file extension
            file_extension = ".jpg" if uploaded_file.type.startswith('image') else ".pdf"
            
            # Create a temporary file with the determined extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file

def display_results(images, text_regions, extracted_text):
    # Display uploaded image or PDF
    st.subheader("Uploaded Image/Document:")
    for img in images:
        st.image(img, caption="Uploaded Image/PDF", use_column_width=True)

    # Display detected text regions
    st.subheader("Text Regions Detected:")
    image_with_boxes = np.array(images[0].copy())
    for box in text_regions:
        x_values = box[:, 0]
        y_values = box[:, 1]
        x1, y1 = int(np.min(x_values)), int(np.min(y_values))
        x2, y2 = int(np.max(x_values)), int(np.max(y_values))
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
    st.image(image_with_boxes, caption="Uploaded Image/PDF", use_column_width=True)

    # Display extracted text
    st.subheader("Extracted Text:")
    st.info("\n".join(extracted_text))
    st.success("OCR process completed successfully")