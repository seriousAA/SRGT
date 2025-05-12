import os
import zipfile
import glob
import shutil

def process_folders(folder_paths):
    """
    Process a list of folder paths containing txt files and a zip file.
    
    1. Check if any txt files are empty
    2. If empty files are found, copy content from non-empty txt files as placeholder
    3. Recreate the zip file if any modifications were made
    """
    for folder_path in folder_paths:
        print(f"Processing folder: {folder_path}")
        
        # Find all txt files in the folder
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        if not txt_files:
            print(f"No txt files found in {folder_path}")
            continue
        
        # Find zip file in the folder
        zip_files = glob.glob(os.path.join(folder_path, "*.zip"))
        if not zip_files:
            print(f"No zip file found in {folder_path}")
            continue
        
        zip_file_path = zip_files[0]  # Assuming there's only one zip file
        
        # Check for empty txt files and track if any modifications are made
        modifications_made = False
        placeholder_content = None
        
        # First, find non-empty txt file to use as placeholder source
        for txt_file in txt_files:
            if os.path.getsize(txt_file) > 0:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    placeholder_content = f.readline().strip()
                    if placeholder_content:
                        break
        
        if placeholder_content is None:
            print(f"Warning: Could not find any non-empty txt file in {folder_path}")
            continue
        
        # Check and fix empty txt files
        for txt_file in txt_files:
            if os.path.getsize(txt_file) == 0:
                print(f"Found empty file: {txt_file}")
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(f"{placeholder_content}\n")
                print(f"Added placeholder content to {txt_file}")
                modifications_made = True
        
        # Recreate zip file if modifications were made
        if modifications_made:
            print(f"Recreating zip file: {zip_file_path}")
            
            # Remove the original zip file
            os.remove(zip_file_path)
            
            # Create a new zip file with the same name
            with zipfile.ZipFile(zip_file_path, 'w') as new_zip:
                for txt_file in txt_files:
                    # Add txt file to the zip with just the filename, not the full path
                    new_zip.write(txt_file, os.path.basename(txt_file))
            
            print(f"Zip file recreated successfully: {zip_file_path}")
        else:
            print(f"No empty txt files found in {folder_path}")
            
        print("-" * 50)

# Example usage
if __name__ == "__main__":
    # Replace this with your list of folder paths
    folders_to_process = [
        "results/partial_mcl_fcos_obb/4_5_mcl_fcos_obb_0423",
        "results/partial_mcl_fcos_obb/6_5_mcl_fcos_obb_0424",
        "results/partial_mcl_fcos_obb/8_5_mcl_fcos_obb_0425",
        "results/partial_mcl_fcos_obb/2_10_mcl_fcos_obb_0427",
        "results/partial_mcl_fcos_obb/4_10_mcl_fcos_obb_0428",
        "results/partial_mcl_fcos_obb/6_10_mcl_fcos_obb_0429",
        "results/partial_mcl_fcos_obb/8_10_mcl_fcos_obb_0430",
        "results/partial_mcl_fcos_obb/10_10_mcl_fcos_obb_0430"
    ]
    
    process_folders(folders_to_process)
