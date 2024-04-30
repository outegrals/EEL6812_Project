from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from text_extraction.extract import Extractor
import os
import numpy as np
import argparse
import pickle

def predict_document(text):
    # Load the model
    classifier_path = os.path.join(os.getcwd(), 'document_classification')
    model_path = os.path.join(classifier_path, 'classifier_model.h5')
    model = load_model(model_path)

    tokenizer_path = os.path.join(classifier_path, 'tokenizer.pickle')
    # Load the tokenizer
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Tokenizing text
    sequence = tokenizer.texts_to_sequences([text])  # Treat the whole text as one sequence
    max_length = 500  # Use a fixed max_length or the same max_length as was used in training
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')

    # Predict
    predictions = model.predict(padded_sequence)
    label_dict = {'historical': 0, 'medical': 1, 'legal': 2}
    predicted_label_index = np.argmax(predictions)
    predicted_label = [key for key, value in label_dict.items() if value == predicted_label_index][0]
    print(f"The predicted category is: {predicted_label}")

if __name__ == "__main__":
    # Create argument parser to parse command line arguments
    parser = argparse.ArgumentParser(description="Predict document type given image")
    parser.add_argument("--image_path", help="Path to the image file from which to extract text.")
    parser.add_argument("--dir", help="Directory containing images to run on")

    # Parse arguments
    args = parser.parse_args()

    if args.dir:
        for file in os.listdir(args.dir):
            print(f"Predicting on {file}")
            file_path = os.path.join(args.dir, file)
            print("Extracted Text:")
            txt = Extractor(file_path)
            print("--------------------------------")
            predict_document(txt)
            print("--------------------------------")
            print()

    # If image path is provided, extract text from image

    if args.image_path:
        print(f"Predicting on {args.image_path}")
        print("Extracted Text:")
        txt = Extractor(args.image_path)
        print("--------------------------------")
        predict_document(txt)
        print("--------------------------------")
        print()
