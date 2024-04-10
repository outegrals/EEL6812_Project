from PIL import Image
import numpy as np
import pytesseract
import numpy as np
import argparse

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

def main():
    # Create argument parser to parse command line arguments
    parser = argparse.ArgumentParser(description="Extract text from an image using Tesseract's LSTM engine.")
    parser.add_argument("image_path", help="Path to the image file from which to extract text.")

    # Parse arguments
    args = parser.parse_args()

    # Extract text from the image at the specified path
    extracted_text = extract_text(args.image_path)

    print(extracted_text)

if __name__ == "__main__":
    main()