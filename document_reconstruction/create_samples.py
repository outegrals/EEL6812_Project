# create samples (positive and negative) for each shredded document
# uses full strips so gives 38 pos and 38 negative samples per document
# can modify to use segments of each strip for sample creation

import os
import random
import numpy as np
from PIL import Image

# create positive and negative samples for training
def create_samples(folder_path, positive_folder, negative_folder, edge_width=15):
    # list of strip files
    strip_files = sorted(os.listdir(folder_path), key=lambda x: int(x.split('_')[2].split('.')[0]))

    # create directories for positive and negative samples if not already
    if not os.path.exists(positive_folder):
        os.makedirs(positive_folder, exist_ok=True)
    if not os.path.exists(negative_folder):
        os.makedirs(negative_folder, exist_ok=True)

    num_pos_samples = 0
    # positive samples (adjacent strips)
    for i in range(len(strip_files) - 1):
        strip1_path = os.path.join(folder_path, strip_files[i])
        strip2_path = os.path.join(folder_path, strip_files[i + 1])

        # open the images
        strip1 = Image.open(strip1_path).convert("L")  # grayscale
        strip2 = Image.open(strip2_path).convert("L")  # grayscale

        # get right edge of strip1 and left edge of strip2
        right_edge_strip1 = strip1.crop((strip1.width - edge_width, 0, strip1.width, strip1.height))
        left_edge_strip2 = strip2.crop((0, 0, edge_width, strip2.height))

        # concatenate edges to create a sample
        positive_sample = Image.new("L", (edge_width * 2, strip1.height))
        positive_sample.paste(right_edge_strip1, (0, 0))
        positive_sample.paste(left_edge_strip2, (edge_width, 0))

        # save the positive sample
        positive_sample.save(os.path.join(positive_folder, f"positive_{folder_path.split('/')[1]}_{i + 1}.png"))
        num_pos_samples +=1

    
    num_neg_samples = 0 # num_negative_samples
    # negative samples (non-adjacent strips)
    while num_neg_samples != num_pos_samples:
        
        # randomly select two non-adjacent strips
        strip1_idx = random.randint(0, len(strip_files) - 1)
        temp_strip = random.randint(0, len(strip_files) - 1)
        if temp_strip != (strip1_idx + 1):
            strip2_idx = temp_strip

            strip1_path = os.path.join(folder_path, strip_files[strip1_idx])
            strip2_path = os.path.join(folder_path, strip_files[strip2_idx])

            # open the images
            strip1 = Image.open(strip1_path).convert("L")
            strip2 = Image.open(strip2_path).convert("L")

            # get right edge of strip1 and left edge of strip2
            right_edge_strip1 = strip1.crop((strip1.width - edge_width, 0, strip1.width, strip1.height))
            left_edge_strip2 = strip2.crop((0, 0, edge_width, strip2.height))

            # concatenate edges to create a negative sample
            negative_sample = Image.new("L", (edge_width * 2, strip1.height))
            negative_sample.paste(right_edge_strip1, (0, 0))
            negative_sample.paste(left_edge_strip2, (edge_width, 0))

            # save the negative sample
            negative_sample.save(os.path.join(negative_folder, f"negative_{folder_path.split('/')[1]}_{num_neg_samples + 1}.png"))
            num_neg_samples += 1
            

positive_folder = 'samples/positive'  # folder for positive samples
negative_folder = 'samples/negative'  # folder for negative samples

# create samples for all documents shredded documents
source_dir = 'strip_shredded_documents'

for filename in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, filename)    # path to the folder with shredded pieces
    if os.path.isdir(folder_path):
        create_samples(folder_path, positive_folder, negative_folder)
            



