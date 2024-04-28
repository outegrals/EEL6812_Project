from PIL import Image
import pytesseract
import numpy as np
import argparse
import cv2
from spellchecker import SpellChecker

def clean_image(image_path):
    original_image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding to get a binary image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(original_image).shape[0]//30))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Inpaint the detected lines on the original image
    return cv2.inpaint(original_image, detected_lines, 3, cv2.INPAINT_TELEA)


def extract_text(input_data):
    # If input is a string, assume it's a file path and open the image
    if isinstance(input_data, str):
        image = Image.open(input_data)
    # If input is a numpy array, convert it to a PIL Image
    elif isinstance(input_data, np.ndarray):
        image = Image.fromarray(input_data)
    else:
        raise ValueError("Unsupported input type. Please provide a file path or a numpy array.")
    
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(image)
    return text

def correct_text(ocr_output):
    spell = SpellChecker()

    # Split the paragraph into individual words
    words = ocr_output.split()

    # Find those words that may be misspelled
    misspelled = spell.unknown(words)

    corrected_text = []
    # Iterate through the words in the paragraph
    for word in words:
        # If the word is misspelled
        if word in misspelled:
            # Get the one 'most likely' correction
            corrected_word = spell.correction(word)
            corrected_text.append(corrected_word)
        else:
            # If the word is correct, keep it as is
            corrected_text.append(word)

    # Join the corrected list of words back into a paragraph and make sure there's not none type in it
    filtered_text = [text for text in corrected_text if text is not None]
    corrected_paragraph = ' '.join(filtered_text)
    
    return corrected_paragraph

def Extractor(image_path):

    img_cleaned = clean_image(image_path)
    # Extract text from the image at the specified path
    extracted_text = extract_text(img_cleaned)

    #correct any misspelled words
    corrected_text = correct_text(extracted_text)

    print(corrected_text)
    return corrected_text

if __name__ == "__main__":
    # Create argument parser to parse command line arguments
    parser = argparse.ArgumentParser(description="Extract text from an image using Tesseract's LSTM engine.")
    parser.add_argument("image_path", help="Path to the image file from which to extract text.")

    # Parse arguments
    args = parser.parse_args()
    
    Extractor(args.image_path)