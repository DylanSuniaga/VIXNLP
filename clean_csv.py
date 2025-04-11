import csv
import re
import os
from tqdm import tqdm

# Paths
input_file = 'clean_macro_5y_news_1kcap.csv'
output_file = 'thoroughly_cleaned_news.csv'

print(f"Cleaning file: {input_file}")

# Count total lines for progress bar
total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
print(f"Total lines in file: {total_lines}")

# Read the header first to know the column structure
with open(input_file, 'r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    header = next(reader)

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
    
    # Write the header
    writer.writerow(header)
    
    # Skip the header row in the input
    next(reader, None)
    
    # Process each row with progress bar
    skipped_rows = 0
    processed_rows = 0
    
    for row in tqdm(reader, total=total_lines-1, desc="Cleaning CSV"):
        # Skip completely empty rows
        if not any(row):
            skipped_rows += 1
            continue
        
        # Clean each field in the row
        cleaned_row = []
        for field in row:
            # Clean strings thoroughly
            if field:
                # Replace tabs with spaces
                field = field.replace('\t', ' ')
                # Replace all newlines with spaces
                field = re.sub(r'[\r\n]+', ' ', field)
                # Replace multiple spaces with a single space
                field = re.sub(r'\s+', ' ', field).strip()
            cleaned_row.append(field)
        
        # Write the cleaned row
        writer.writerow(cleaned_row)
        processed_rows += 1

print(f"Cleaning complete:")
print(f"- Processed rows: {processed_rows}")
print(f"- Skipped empty rows: {skipped_rows}")
print(f"Thoroughly cleaned file saved as {output_file}") 