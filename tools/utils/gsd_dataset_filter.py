import pandas as pd
from PIL import Image
import os

def is_mostly_dark(image_path, threshold=(10, 10, 10), dark_ratio=0.8):
    """
    Check if more than dark_ratio of the pixels in the image are below the given threshold.
    Handles both grayscale ('L') and RGB images appropriately.
    """
    image = Image.open(image_path)
    pixels = list(image.getdata())
    
    if image.mode == 'L':  # Grayscale image
        # For grayscale, the threshold is the average of the provided RGB threshold
        gray_threshold = sum(threshold) / 3
        dark_pixels = sum(1 for pixel in pixels if pixel < gray_threshold)
    else:  # RGB or other color image
        # Convert to RGB to ensure consistency for non-grayscale images
        if image.mode != 'RGB':
            image = image.convert('RGB')
            pixels = list(image.getdata())
        dark_pixels = sum(1 for pixel in pixels if all(channel < t for channel, t in zip(pixel, threshold)))

    return dark_pixels / len(pixels) > dark_ratio

def filter_and_remove_images(folder_path, csv_path):
    """
    Filter out images that are mostly dark, remove their entries from the CSV, and delete the images from the folder.
    """
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_path)

    # Initialize an empty list to hold the indices of rows to keep
    indices_to_keep = []

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        image_path = os.path.join(folder_path, row['image name']+'.png')
        # Check if the image is not mostly dark
        if not is_mostly_dark(image_path):
            # If not, add its index to the list of indices to keep
            indices_to_keep.append(index)
        else:
            # If it is mostly dark, delete the image file
            os.remove(image_path)
            print(f"Removed dark image: {row['image name']}")

    # Filter the DataFrame to keep only the rows with indices in indices_to_keep
    filtered_df = df.loc[indices_to_keep]

    # Save the filtered DataFrame to a new CSV file
    new_csv_path = os.path.splitext(csv_path)[0] + '_filtered.csv'
    filtered_df.to_csv(new_csv_path, index=False)
    print(f"Filtered CSV saved to {new_csv_path}")

# Example usage
folder_path = '/home/liyuqiu/RS-PCT/data/DOTA/gsd_dataset/trainval'  # Update this path
csv_path = '/home/liyuqiu/RS-PCT/data/DOTA/gsd_dataset/csv/trainval_data.csv'  # Update this path
filter_and_remove_images(folder_path, csv_path)
