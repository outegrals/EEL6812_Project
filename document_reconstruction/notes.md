# Notes for document_reconstruction/
used https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/isri-ocr-evaluation-tools/legal.4B.tar.gz 
from https://code.google.com/archive/p/isri-ocr-evaluation-tools/downloads?page=2
for testing so far

- filter_documents.py saves all .tif files to separate folder 'documents_to_shred'
- shredder.py shreds the documents in 'documents_to_shred' and saves the pieces for each document in its own folder
can choose between strip shredded or cross-cut shredded
- create_samples.py creates positive and negative samples for testing, 38 of each using the entire strip, saves pos and neg samples in separate files
- display_strip.py and display_crosscut.py currently reorder the strips given their labels, will need to be adjusted to be used for displaying NN results
