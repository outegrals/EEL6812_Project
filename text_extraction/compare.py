from extract import Extractor
import os
import argparse
import matplotlib.pyplot as plt

# Function to compare two texts and plot the results 
def compare_texts_and_plot(text1, text2):
    # Normalize the texts by converting them to lower case
    text1, text2 = text1.lower(), text2.lower()
    
    # Tokenize the texts into words
    words1, words2 = set(text1.split()), set(text2.split())
    print(f'Total words extracted from first image: {len(words1)}')
    print(f'Total words extracted from Second image: {len(words2)}')
    
    # Use set intersection to find common words
    common_words = words1.intersection(words2)
    unique_words = words2.difference(words1)
    matched_count = len(common_words)
    
    # Calculate non-matched words in the first sentence
    non_matched_count = len(unique_words)
    
    print(f"Total Matched Words: {matched_count}")
    
    print(f"Unique words: {unique_words}")
    
    # Calculate similarity as the ratio of common words to total words in the second sentence
    if len(words2) > 0:
        similarity_percentage = (len(common_words) / len(words2)) * 100
        unique_percentage = (len(unique_words) / len(words2)) * 100
    else:
        similarity_percentage, unique_percentage = 0, 0  # Handle the case where the second sentence is empty

    # Data to plot
    labels = ['Text extracted']
    common_data = [similarity_percentage]
    unique_data = [unique_percentage]

    # Plotting the stacked bar chart
    plt.figure(figsize=(7, 5))
    plt.bar(labels, common_data, width=0.25, color='blue', label='Matched Words')
    plt.bar(labels, unique_data, width=0.25, bottom=common_data, color='red', label='Unique Words')
    plt.ylabel('Percentage')
    plt.title('Word Match Percentage')
    plt.legend()
    plt.ylim(0, 100)  # Ensure y-axis starts at 0 and ends at 100%
    plt.show()


if __name__ == "__main__":
    # Create argument parser to parse command line arguments
    parser = argparse.ArgumentParser(description="Compare text between two images")
    parser.add_argument("image_path_1", help="Path to the image file from which to extract text.")
    parser.add_argument("image_path_2", help="Path to the image file from which to extract text.")

    # Parse arguments
    args = parser.parse_args()

    txt1 = Extractor(args.image_path_1)
    txt2 = Extractor(args.image_path_2)
    
    print("-----------------------------------------------")
    
    compare_texts_and_plot(txt1, txt2)