import os
import pandas as pd

def match_images_with_csv(folder_path, csv_file_path, output_csv_file_path):
    # Load the CSV file containing image names and GSD values
    data = pd.read_csv(csv_file_path)
    
    # List all PNG files in the given folder and remove the extension to match the CSV format
    image_files = {os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith('.png')}
    
    # Create a set of image names from the CSV for fast lookup
    csv_image_names = set(data['image name'])
    
    # Find matched and unmatched images using set operations
    matched_images = image_files & csv_image_names
    unmatched_images = image_files - csv_image_names
    
    # Save matched entries to a new CSV file
    if matched_images:
        matched_data = data[data['image name'].isin(matched_images)]
        matched_data.to_csv(output_csv_file_path, index=False)
        print(f"Matched entries saved to {output_csv_file_path}")
    
    # Report unmatched image names
    if unmatched_images:
        print("The following images did not have matching entries in the CSV:")
        for img in unmatched_images:
            print(img)
    else:
        print("All images have corresponding entries in the CSV.")

# Example usage:
folder_path = '/home/liyuqiu/RS-PCT/data/DOTA/gsd_dataset/test'
csv_file_path = '/home/liyuqiu/RS-PCT/data/DOTA/gsd_dataset/csv/test_data_filtered.csv'
output_csv_file_path = '/home/liyuqiu/RS-PCT/data/DOTA/gsd_dataset/csv/test_data_no_difficult.csv'
match_images_with_csv(folder_path, csv_file_path, output_csv_file_path)
