# EEL6812_Project
Final Project For EEL 6812 Intro To Deep learning 

Author: Zoe Bats, Julio Enrique Marquez, Tommy Truong

# Project Overview

The purpose of this project is to be able to take in any type of shredded image and reconstructed in order to analysis what it is.

# Dependencies
    - Python (3.7)
    - Argparse
    - PyTorch
    - Scikit-learn
    - NumPy
    - Matplotlib
    - Pillow
    - Pytesseract
    - OpenCV2
    - pyspellchecker

# Note
PyTesseract is a wrapper to tesseract.

Tesseract can be found here:
https://tesseract-ocr.github.io/tessdoc/Installation.html

If on windows, after installing Tesseract OCR from the above link
add this in your system environment variable : C:\Program Files\Tesseract-OCR\tesseract

Also note, document reconstruction has it's own README to follow

# Run

Document Reconstruction
=======================
```bash
usage: reconstruct_model.py [-h]

reconstructs a shredded image
```

Text Extraction
===============
```bash
usage: extract.py [-h] image_path

Extract text from an image using Tesseract's LSTM engine.

positional arguments:
  image_path  Path to the image file from which to extract text.
```

Text Comparision
================
```bash
usage: compare.py [-h] image_path_1 image_path_2

Compare text between two images

positional arguments:
  image_path_1  Path to the image file from which to extract text.
  image_path_2  Path to the image file from which to extract text.
```
Document Classification
=======================
```bash
usage: classify.py [-h] image_path

Classify a document using a trained model
```