# legal.4B downloaded from https://code.google.com/archive/p/isri-ocr-evaluation-tools/downloads?page=2 
# saves all .tif files to a new folder called 'documents_to_shred'
import os
import shutil

# Source and destination directories
source_dir = 'documents/legal.4B'
destination_dir = 'documents_to_shred'

# Create destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Function to gather .tif files from subdirectories and move them
def gather_and_move_tif_files(source_dir, destination_dir):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".tif"):
                file_path = os.path.join(root, file)
                shutil.copy(file_path, destination_dir)

# Call the function to gather and move .tif files
gather_and_move_tif_files(source_dir, destination_dir)
