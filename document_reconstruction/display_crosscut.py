# reconstruct and display labeled cross shredded pieces 
# will need to be modified to accept order given by NN
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Function to piece together cross-cut shredded documents
def piece_together_cross_cut_document(folder_path, show_image=True):
    # Get list of shredded pieces
    strip_files = sorted(os.listdir(folder_path), key=lambda x: (int(x.split('_')[2]), int(x.split('_')[3].split('.')[0])))  # Sorting by vertical then horizontal

    # Organize the strips into a dictionary by their vertical position
    strips_by_vertical_position = {}
    for file in strip_files:
        strip_path = os.path.join(folder_path, file)
        vertical_pos = int(file.split('_')[2])
        if vertical_pos not in strips_by_vertical_position:
            strips_by_vertical_position[vertical_pos] = []
        strips_by_vertical_position[vertical_pos].append(strip_path)

    # Calculate document dimensions based on the first strip
    first_strip = Image.open(strips_by_vertical_position[1][0])
    strip_width = first_strip.width
    strip_height = first_strip.height

    # Determine the full document dimensions
    num_vertical = len(strips_by_vertical_position)  # Number of vertical strips
    num_horizontal = len(strips_by_vertical_position[1])  # Number of horizontal strips
    doc_width = strip_width * num_vertical
    doc_height = strip_height * num_horizontal

    # Create a blank canvas to assemble the document
    reconstructed_image = Image.new('L', (doc_width, doc_height))  # 'L' for grayscale

    # Paste each strip in its correct position
    for vertical_pos, strips in strips_by_vertical_position.items():
        for horizontal_pos, strip_path in enumerate(strips, start=1):
            strip = Image.open(strip_path)
            position = ((vertical_pos - 1) * strip_width, (horizontal_pos - 1) * strip_height)
            reconstructed_image.paste(strip, position)

    # Display the reconstructed document
    if show_image:
        plt.imshow(reconstructed_image, cmap='gray')
        plt.axis('off')
        plt.title('Reconstructed Cross-Cut Document')
        plt.show()

    return reconstructed_image

# Example usage
folder_path = 'crosscut_shredded_documents/9410_005.4B'  # Path to the folder with shredded pieces
reconstructed_document = piece_together_cross_cut_document(folder_path)




