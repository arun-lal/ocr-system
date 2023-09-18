from pdf2image import convert_from_path
from PIL import Image
import numpy as np
from trocr import TrOCRProcessor, VisionEncoderDecoderModel
from craft_text_detector import load_craftnet_model, load_refinenet_model, get_prediction
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRProcessor:
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
    
    def process_file(self, input_type, file_path):
        images = []
        text_regions = []
        extracted_text = []

        if input_type == "Upload Image/PDF":
            try:
                actual_file_extension = file_path.split('.')[-1].lower()
                if actual_file_extension in ["jpg", "jpeg"]:
                    image = self.process_image(file_path)
                    images.append(image)
                elif actual_file_extension == "pdf":
                    pdf_images = self.process_pdf(file_path)
                    images.extend(pdf_images)

                if images:
                    text_regions = self.perform_text_detection(np.array(images[0]))
                    extracted_text = self.extract_text_from_regions(images[0], text_regions)

            except Exception as e:
                st.error(f"Error processing the uploaded file: {e}")

        return images, text_regions, extracted_text

