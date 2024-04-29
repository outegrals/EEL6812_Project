# Notes for document_reconstruction/
used https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/isri-ocr-evaluation-tools/legal.4B.tar.gz 
from https://code.google.com/archive/p/isri-ocr-evaluation-tools/downloads?page=2
for testing so far

# Order of scripts to run
- filter_documents.py
- shredder.py
- create_samples.py
- reconstruct_model.py
- display_strip.py (optional)
- display_crosscut.py (optional)

# Notes for document_reconstruction/

# filter_documents.py 
- saves all .tif files to separate folder 'documents_to_shred'
  
# shredder.py 
- shreds the documents in 'documents_to_shred' and saves the pieces for each document in its own folder
- can choose between strip shredded or cross-cut shredded
- can adjust the number of strips and cross cuts
- each strip is saved as 'shredded_strip_1' with number corresponding to its place
- cross cut shreds are saved as 'shredded_strip_1_2' with the first number representing to the column and the second number representing the row
- ex. 'shredded_strip_2_3' will be to the right of 'shredded_strip_1_3' and above 'shredded_strip_2_4'
  
# create_samples.py
- creates positive and negative samples for testing, 38 of each using the entire strip, saves pos and neg samples in separate files in 'samples' folder
- takes a little while to run
  
# display_strip.py and display_crosscut.py 
- currently reorder the strips given their labels
- will need to be adjusted to be used for displaying NN results


