# reconstruct and display strip shredded document given correctly labeled pieces
# will need to be modified to accept output from NN
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# function to piece together shredded strips
def piece_together_shredded_document(folder_path, strip_width=None, show_image=True):
    # get list of shredded pieces
    strip_files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sorting by number
    strips = [Image.open(os.path.join(folder_path, file)) for file in strip_files]

    # Determine the width of the full document based on the number of strips and their width
    if strip_width is None:
        strip_width = strips[0].width

    # Height of the document is the height of the first strip
    doc_height = strips[0].height
    doc_width = strip_width * len(strips)

    # Create a blank canvas to assemble the document
    reconstructed_image = Image.new('L', (doc_width, doc_height))  # 'L' for grayscale

    # Paste each strip in its correct position
    for i, strip in enumerate(strips):
        position = (i * strip_width, 0)
        reconstructed_image.paste(strip, position)

    # Display the reconstructed document
    if show_image:
        plt.imshow(reconstructed_image, cmap='gray')
        plt.axis('off')
        plt.title('Reconstructed Document')
        plt.show()

    return reconstructed_image


folder_path = 'strip_shredded_documents/9410_005.4B'  # Path to the folder with shredded pieces
reconstructed_document = piece_together_shredded_document(folder_path)
