"""
This script partitions a given .pkl file containing image annotations into labeled and unlabeled datasets based on 
a specified label rate. The 'content' list in the .pkl file is sampled to create two separate .pkl files for labeled 
and unlabeled data. The partitioning ensures that each category is represented in the labeled data at least once in 
at least 10% of its entries, which is crucial for the effectiveness of semi-supervised learning.

Parameters:
- input_file (str): Path to the input .pkl file containing the 'cls' and 'content' keys. 'cls' remains the same in 
  both outputs, while 'content' is partitioned.
- output_folder (str): Base directory path where the output .pkl files will be stored. The directory structure will 
  include subdirectories named as '{fold_id}_{label_rate}'.
- label_rate (int): The percentage of the data to be labeled (e.g., 1 for 1%, 2 for 2%).
- fold_id (int): The index of the current partition fold.

Output Structure:
Each fold results in two .pkl files:
- labeled_ann.pkl: Contains the labeled subset of 'content'.
- unlabeled_ann.pkl: Contains the unlabeled subset of 'content'.

The script ensures that every category (from 0 to 17) appears at least in the minimum specified percentage of the 
labeled datasets. Adjustments are made upon failure to meet the conditions, ensuring compliance with requirements 
for semi-supervised learning.

Usage:
Run the script from the command line with the necessary arguments:
    python partition_script.py <input_file_path> <label_rate> <fold_id> <output_folder_path>

Example:
    python partition_script.py 'path/to/annotations.pkl' 1 5 'path/to/output/folder'

Note:
The script will terminate the partitioning process if it fails to meet the category coverage conditions after 
reducing the minimum percentage coverage to as low as 1%.
"""

import os
import pickle
import random
from collections import defaultdict
import datetime
import logging
import os.path as osp
from multiprocessing import Pool, Manager

def setup_logger(log_path):
    logger = logging.getLogger('img split')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = osp.join(log_path, now + '.log')
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, 'w')
    ]

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def parse_categories(content_list):
    category_data = defaultdict(list)
    for index, content in enumerate(content_list):
        categories = set(content['ann']['labels'])
        for category in categories:
            category_data[category].append(index)
    return category_data

def prepare_datasets(input_file, output_folder, label_rate, fold_id):
    data = load_pickle(input_file)
    content = data['content']
    total_items = len(content)
    num_labeled = max(1, total_items * label_rate // 100)
    fold_dir = os.path.join(output_folder, f'{fold_id}_{label_rate}')
    os.makedirs(fold_dir, exist_ok=True)
    logger = setup_logger(fold_dir)
    manager = Manager()
    lock = manager.Lock()
    logger.info(f"Preparing fold {fold_id} with label rate {label_rate}...")
    random.seed(fold_id)
    min_category_percentage = 16
    success = False
    category_data = parse_categories(content)

    while min_category_percentage >= 1:
        for _ in range(3):  # Retry sampling up to 3 times
            all_selected_labeled = set()
            all_selected_unlabeled = set()
            required_category_coverage = max(1, (num_labeled * min_category_percentage) // 10000)

            # Ensure each category is covered sufficiently in the labeled set
            for _, indices in category_data.items():
                indices = set(indices)
                num_intersect_labeled = len(indices & all_selected_labeled)
                indices_for_labeled = list(indices-all_selected_unlabeled)
                random.shuffle(indices_for_labeled)
                sample_len_labeled = max(0, min(required_category_coverage,
                                         int(len(indices)*0.5)) - num_intersect_labeled)
                all_selected_labeled.update(indices_for_labeled[:sample_len_labeled])
                num_intersect_unlabeled = len(indices & all_selected_unlabeled)
                indices_for_unlabeled = list(indices-all_selected_labeled)
                random.shuffle(indices_for_unlabeled)
                sample_len_unlabeled = max(0, int(len(indices)*0.5) - num_intersect_unlabeled)
                # print(f"{sample_len_labeled} {sample_len_unlabeled} {len(indices)}")
                all_selected_unlabeled.update(indices_for_unlabeled[:sample_len_unlabeled])
                # print(len(all_selected_labeled & all_selected_unlabeled))
                if len(all_selected_labeled) > num_labeled:
                    break

            if len(all_selected_labeled) > num_labeled:
                continue
            else:
                # Add more data randomly if needed
                remaining_indices = set(range(total_items)) - (all_selected_labeled | all_selected_unlabeled)
                additional_data_needed = num_labeled - len(all_selected_labeled)
                if additional_data_needed > 0 and len(remaining_indices) >= additional_data_needed:
                    all_selected_labeled.update(random.sample(remaining_indices, 
                                                                additional_data_needed))
                elif len(remaining_indices) < additional_data_needed:
                    continue
                success = True
                break

        if success:
            break
        else:
            # lock.acquire()
            min_category_percentage -= 1  # Reduce minimum percentage by 1%
            logger.info(f"Reducing category coverage requirement to {min_category_percentage}% and retrying...")
            # lock.release()

    if not success:
        # lock.acquire()
        logger.info(f"Failed to prepare fold {fold_id} with sufficient category coverage. Terminating...")
        # lock.release()
        return

    # Split into labeled and unlabeled sets
    labeled_indices = list(all_selected_labeled)
    unlabeled_indices = list(set(range(total_items)) - all_selected_labeled)

    labeled_content = [content[i] for i in labeled_indices]
    unlabeled_content = [content[i] for i in unlabeled_indices]

    # Save the results
    labeled_folder = os.path.join(fold_dir, 'labeled')
    unlabeled_folder = os.path.join(fold_dir, 'unlabeled')
    os.makedirs(labeled_folder, exist_ok=True)
    os.makedirs(unlabeled_folder, exist_ok=True)
    save_pickle({'cls': data['cls'], 'content': labeled_content}, 
                os.path.join(labeled_folder, 'patch_annfile.pkl'))
    save_pickle({'cls': data['cls'], 'content': unlabeled_content}, 
                os.path.join(unlabeled_folder, 'patch_annfile.pkl'))
    # lock.acquire()
    logger.info(f"Fold {fold_id} with label rate {label_rate} prepared, "
                f"labeled {len(labeled_indices)} items, unlabeled {len(unlabeled_indices)} items "
                f"and labeled \u2229 unlabeled {len(all_selected_labeled & all_selected_unlabeled)} items.")
    # lock.release()

if __name__ == "__main__":
    import sys
    input_file_path = sys.argv[1]
    label_rate = int(sys.argv[2])
    fold_id = int(sys.argv[3])
    output_folder_path = sys.argv[4]
    prepare_datasets(input_file_path, output_folder_path, label_rate, fold_id)
