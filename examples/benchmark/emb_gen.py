import os
import sys
import csv

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from lib.text import encode as encode_text

def process_queries(input_file="queries.txt", output_file="vector.csv"):
    """
    Reads queries from input_file, encodes them, and saves them to output_file in CSV format.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)

        for line in infile:
            query_text = line.strip()
            if not query_text:  # Skip empty lines
                continue
            import pdb;pdb.set_trace()
            embs = encode_text(query_text, query=True)
            # Convert embedding to a string representation if it's a list/array.
            # A common way is to join list elements with a semicolon or other delimiter,
            # or store as a JSON string. For simplicity, let's convert to string directly.
            embedding_str = str(embs[0]) 
            csv_writer.writerow([query_text, embedding_str])

if __name__ == "__main__":
    process_queries()
    print("Processing complete. Output saved to vector.csv")
