# shred documents (strip or cross cut) can change shred_type in main
# can change num_strips and num_cuts for both types in shred_document
import cv2
import os
import numpy as np
import random
import shutil
import argparse

def shred_document(image_path, output_folder, noise_factor, shred_type):
    
    # load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    
    # Create subfolder for shredded pieces
    document_name = os.path.splitext(os.path.basename(image_path))[0]
    document_folder = os.path.join(output_folder, document_name)
    if not os.path.exists(document_folder):
        os.makedirs(document_folder)
    
    # determine shred type (based on 8.5" x 11" paper)
    if shred_type == 'strip':
        num_strips = 39  # number of strips to shred the document into (typical strip shredder approx 39 strips)
        
        # calculate strip width
        strip_width = width // num_strips
        
        # shred the document
        for i in range(num_strips):
            start_x = i * strip_width
            end_x = (i + 1) * strip_width
            shredded_vertical = image[:, start_x:end_x]

             # add noise to the edges
            noise = np.random.normal(0, noise_factor * 255, size=(height, 2)).astype(np.uint8)
            shredded_vertical[:, :2] += noise
            shredded_vertical[:, -2:] += noise
            shredded_vertical = np.clip(shredded_vertical, 0, 255)

            # save shredded strip
            cv2.imwrite(os.path.join(document_folder, f'shredded_strip_{i+1}.png'), shredded_vertical)
    
    if shred_type == 'crosscut':
        num_strips = 54  # number of strips to shred the document into (typical cross cut shredder approx 54 strips)
        num_cuts = 5  # number of cross cuts per strip (typical for cross cut shredder approx 5)
        
        # calculate piece width and height
        strip_width = width // num_strips
        strip_height = height // num_cuts 
        
        # shred the document (vertically)
        for i in range(num_strips):
            start_x = i * strip_width
            end_x = (i + 1) * strip_width
            shredded_vertical = image[:, start_x:end_x]
            
            # cross cut strips (horizontally)
            for j in range(num_cuts):
                start = j * strip_height
                end = (j + 1) * strip_height
                shredded_horizontal = shredded_vertical[start:end, :]

                # add noise to the edges
                # vertical edges
                noise = np.random.normal(0, noise_factor * 255, size=(strip_height, 2)).astype(np.uint8)
                shredded_horizontal[:, :2] += noise
                shredded_horizontal[:, -2:] += noise
                
                # horizontal edges
                noise = np.random.normal(0, noise_factor * 255, size=(2, strip_width)).astype(np.uint8)
                shredded_horizontal[:2, :] += noise
                shredded_horizontal[-2:, :] += noise
                shredded_horizontal = np.clip(shredded_horizontal, 0, 255)

                # Save shredded piece
                cv2.imwrite(os.path.join(document_folder, f'shredded_strip_{i+1}_{j+1}.png'), shredded_horizontal)

if __name__ == "__main__":
    # Create argument parser to parse command line arguments
    parser = argparse.ArgumentParser(description="Shreds an input document")
    parser.add_argument("--shred_type", default="strip", help="Shred type (crosscut or strips)")
    parser.add_argument("--source_dir", default="documents_to_shred", help="Path containing documents to shred")
    parser.add_argument("--dest_dir", default=None, help="Path to save shredded documents")
    parser.add_argument("--noise", default=0.1, help="Path to save shredded documents")

    args = parser.parse_args()
    if args.dest_dir is None:
        args.dest_dir = f"{args.shred_type}_shredded_documents"

    print("Shred Type:", args.shred_type)
    print("Destination Directory:", args.dest_dir)
    print("Noise Factor: ", args.noise)

    # shred type
    shred_type = args.shred_type
    noise_factor = float(args.noise)
    # source and destination directories
    source_dir = args.source_dir
    destination_dir = args.dest_dir
    
    # create destination if doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # shred each document and save in subfolder
    for filename in os.listdir(source_dir):
        image_path = os.path.join(source_dir, filename)
        if os.path.isfile(image_path):
            shred_document(image_path, destination_dir, noise_factor, shred_type)
