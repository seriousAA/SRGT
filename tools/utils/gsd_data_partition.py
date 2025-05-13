import os
import shutil
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split

# User-provided paths
# Example path, needs to be replaced with actual image folder path
image_folder_path = '/home/****/RS-PCT/data/DOTA/gsd_after'  
image_folder_path_xview = '/home/****/RS-PCT/data/DOTA/gsd_xview_after'  
# Example path, needs to be replaced with actual results folder path
results_folder_path = '/home/****/RS-PCT/data/DOTA/gsd_dataset'  

# Load the CSV file
file_path = '/home/****/RS-PCT/gsd_data_after.csv'
file_path_xview = '/home/****/RS-PCT/gsd_xview_after.csv'
gsd_data = pd.read_csv(file_path)
gsd_data_xview = pd.read_csv(file_path_xview)

def get_split_data(gsd_data, high_gsd_threshold):
    # Splitting data with GSD > 10 for test set
    high_gsd_data = gsd_data[gsd_data['GSD'] > high_gsd_threshold]
    other_data = gsd_data[gsd_data['GSD'] <= high_gsd_threshold]

    # Split high GSD data into two halves
    high_gsd_test, high_gsd_train_val = train_test_split(high_gsd_data, test_size=0.5, random_state=42)
    lower_test_rate = max((0.15*len(gsd_data)-len(high_gsd_test)), 0.)/len(other_data)
    lower_gsd_train_val, lower_gsd_test = train_test_split(other_data, test_size=lower_test_rate, random_state=42)

    # Combining half of high GSD data with other data for train and val split
    combined_train_val_data = pd.concat([high_gsd_train_val, lower_gsd_train_val])
    test_data = pd.concat([high_gsd_test, lower_gsd_test])
    return combined_train_val_data, test_data

trainval_data, test_data = get_split_data(gsd_data, 10)
trainval_data_xview, test_data_xview = get_split_data(gsd_data_xview, 3)

# Combining trainval and xview data
combined_train_val_data = pd.concat([trainval_data, trainval_data_xview])
combined_test_data = pd.concat([test_data, test_data_xview])

# Splitting combined data into train and val
# train_data, val_data = train_test_split(combined_train_val_data, test_size=0.15/0.85, random_state=42)

# Creating sub-folders in the results folder
subfolders = ['trainval', 'test', 'csv']
for folder in subfolders:
    os.makedirs(os.path.join(results_folder_path, folder), exist_ok=True)

# Function to copy images to respective folders using multithreading
def copy_image(image_folder_path, img_name, subset_name):
    src_path = os.path.join(image_folder_path, img_name + '.png')
    dst_path = os.path.join(results_folder_path, subset_name, img_name + '.png')
    if os.path.exists(src_path):  # Check if the source file exists
        shutil.copy(src_path, dst_path)

def copy_images_multithreaded(image_folder_path, data, subset_name):
    with ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(copy_image, [image_folder_path]*len(data), data['image name'], [subset_name]*len(data))

# Copying images and saving CSV files for each subset
copy_images_multithreaded(image_folder_path, trainval_data, 'trainval')
copy_images_multithreaded(image_folder_path_xview, trainval_data_xview, 'trainval')
combined_train_val_data.to_csv(os.path.join(results_folder_path, 'csv', 'trainval_data.csv'), index=False)
# copy_images_multithreaded(train_data, 'train')
# train_data.to_csv(os.path.join(results_folder_path, 'csv', 'train_data.csv'), index=False)

# copy_images_multithreaded(val_data, 'val')
# val_data.to_csv(os.path.join(results_folder_path, 'csv', 'val_data.csv'), index=False)

copy_images_multithreaded(image_folder_path, test_data, 'test')
copy_images_multithreaded(image_folder_path_xview, test_data_xview, 'test')
combined_test_data.to_csv(os.path.join(results_folder_path, 'csv', 'test_data.csv'), index=False)

print("Script executed. Trainval, and test subsets created along with respective CSV files.")
