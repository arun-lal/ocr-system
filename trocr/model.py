import logging
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
from pdf2image import convert_from_path
import streamlit as st
import cv2
import tempfile
import numpy as np
import time

# import craft functions
from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# image_path = r"C:\Users\pc\Downloads\HSKL_LOGO_RGB_pos (002)_single_line.jpg"
# img_as_array = cv2.imread(image_path)
# # image = Image.open(image_path).convert("RGB")
# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-stage1')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-stage1')

# pixel_values = processor(img_as_array, return_tensors="pt").pixel_values
# generated_ids = model.generate(pixel_values, pad_token_id=processor.tokenizer.eos_token_id)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print("generated_text:", generated_text)

# set image path and export folder directory
image = r"C:\Users\pc\Downloads\1693843206779.jpg" # can be filepath, PIL image or numpy array
output_dir = 'outputs/'

# read image
# image = read_image(image)

# # load models
# refine_net = load_refinenet_model(cuda=False)
# craft_net = load_craftnet_model(cuda=False)

# # perform prediction
# prediction_result = get_prediction(
#     image=image,
#     craft_net=craft_net,
#     refine_net=refine_net,
#     text_threshold=0.7,
#     link_threshold=0.4,
#     low_text=0.4,
#     cuda=False,
#     long_size=1280,
#     poly=False
# )

# # export detected text regions
# exported_file_paths = export_detected_regions(
#     image=image,
#     regions=prediction_result["boxes"],
#     output_dir=output_dir,
#     rectify=True
# )

# # export heatmap, detection points, box visualization
# export_extra_results(
#     image=image,
#     regions=prediction_result["boxes"],
#     heatmaps=prediction_result["heatmaps"],
#     output_dir=output_dir
# )

# unload models from gpu
# empty_cuda_cache()

class OCRModel:
    def __init__(self):
        try:
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-stage1')
            self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-stage1')
            # Load CRAFT models
            self.refine_net = load_refinenet_model(cuda=False)
            self.craft_net = load_craftnet_model(cuda=False)
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def process_image(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise

    def process_pdf(self, pdf_path):
        try:
            images = convert_from_path(pdf_path)
            processed_images = [img.convert("RGB") for img in images]
            return processed_images
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    def perform_text_detection(self, image):
        try:
            # Perform text detection
            prediction_result = get_prediction(
                image=image,
                craft_net=self.craft_net,
                refine_net=self.refine_net,
                text_threshold=0.7,
                link_threshold=0.4,
                low_text=0.4,
                cuda=False,
                long_size=1280,
                poly=False
            )
            return prediction_result["boxes"]
        except Exception as e:
            logger.error(f"Error during text detection: {e}")
            raise
    
    def extract_text_from_regions(self, image, regions):
        extracted_text = []

        try:
            print("Entered extract_text_from_regions...")
            for region in regions:
                x_values = region[:, 0]  # Extract all x coordinates
                y_values = region[:, 1]  # Extract all y coordinates

                # Calculate x1, y1, x2, y2
                x1, y1 = int(np.min(x_values)), int(np.min(y_values))
                x2, y2 = int(np.max(x_values)), int(np.max(y_values))
                # print(f"region: {region}")
                # x1, y1, x2, y2 = region
                print(f"x1: {x1}, y1:{y1}, x2: {x2}, y2:{y2}")

                print("slicing ...")
                # Ensure that the indices are integers
                print("getting x1, ....")
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                print(f"x1: {x1}, y1:{y1}, x2: {x2}, y2:{y2}")
                image_array = np.array(image)

                print("cropping image ...")
                cropped_image = image_array[y1:y2, x1:x2]
                print("Performing ocr...")
                ocr_results = self.perform_ocr([cropped_image])
                extracted_text.extend(ocr_results)

            return extracted_text
        except Exception as e:
            logger.error(f"Error during text detection: {e}")
            raise

    def perform_ocr(self, images):
        try:
            results = []

            for image in images:
                pixel_values = self.processor(image, return_tensors="pt").pixel_values
                generated_ids = self.model.generate(pixel_values, pad_token_id=self.processor.tokenizer.eos_token_id)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                results.append(generated_text)
            
            return results
        except Exception as e:
            logger.error(f"Error during OCR processing: {e}")
            raise

# Define Streamlit app
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

    ocr_model = OCRModel()
    
    # Select input type
    input_type = st.sidebar.radio("Select input type:", ("Upload Image/PDF", "Enter File Path"))
    file_path = None  # Initialize file_path variable

    # Upload an input file
    if input_type == "Upload Image/PDF":
        st.sidebar.markdown("### Upload Image/PDF")

        uploaded_file = st.sidebar.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])
        if uploaded_file is not None:
            # Determine the file extension
            file_extension = ".jpg" if uploaded_file.type.startswith('image') else ".pdf"

            # Create a temporary file with the determined extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(uploaded_file.read())
                file_path = temp_file.name
                print("file_path:", file_path)

            # Determine the actual file type from the file extension
            actual_file_extension = file_path.split('.')[-1].lower()

    else:
        file_path = st.text_input("Enter the file path:")

    # Process the selected file
    if file_path:
        try:
            images = []
            if actual_file_extension in ["jpg", "jpeg"]:            # if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                image = ocr_model.process_image(file_path)
                images.append(image)
            elif actual_file_extension == "pdf":
                # file_path.lower().endswith('.pdf'):
                pdf_images = ocr_model.process_pdf(file_path)
                images.extend(pdf_images)

            if images:

                # Create text extraction progress bar
                extraction_progress_bar = st.empty()
                extraction_progress_text = st.empty()

                # Check if text detection is completed
                if detection_progress_bar.progress(100):
                    # Update extraction progress text
                    extraction_progress_text.text("Extracting text...")

                # Simulate text extraction and update extraction progress bar accordingly
                for percent_complete in range(1, 101):
                    time.sleep(0.05)  # Simulate some processing time
                    extraction_progress_bar.progress(percent_complete / 100)  # Update progress bar
                # Extract text from regions using TROCR
                extracted_text = ocr_model.extract_text_from_regions(images[0], text_regions)

                st.sidebar.subheader("Uploaded Image/Document:")

                for img in images:
                    st.sidebar.image(img, caption="Uploaded Image/PDF", use_column_width=True)

                st.sidebar.subheader("Text Regions detected:")

                # Display input image with bounding boxes
                image_with_boxes = np.array(images[0].copy())
                for box in text_regions:
                    x_values = box[:, 0]  # Extract all x coordinates
                    y_values = box[:, 1]  # Extract all y coordinates
                    
                    # Calculate x1, y1, x2, y2
                    x1, y1 = int(np.min(x_values)), int(np.min(y_values))
                    x2, y2 = int(np.max(x_values)), int(np.max(y_values))

                    # Ensure that the indices are integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
                st.sidebar.image(image_with_boxes, caption="Uploaded Image/PDF", use_column_width=True)

                # st.sidebar.subheader("Extracted Text:")

                # for result in extracted_text:
                #     st.sidebar.write(result)
                
                # Display extracted text in the middle of the page in a box
                st.subheader("Extracted Text:")
                st.info("\n".join(extracted_text))
                st.success("OCR process completed successfully")

                logger.info("OCR process completed successfully")
            else:
                st.warning("No images found for processing.")
        
        except Exception as e:
            logger.error(f"Error during OCR process: {e}")
            st.error(f"Error during OCR process: {e}")

if __name__ == "__main__":
    main()