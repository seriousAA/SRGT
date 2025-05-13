import os
import random

def file_sample(src_path, dst_path, num):
    files=os.listdir(src_path)
    len_num=len(files)
    sample_files_idx=random.sample(range(len_num), num)
    for file_idx in sample_files_idx:
        file=files[file_idx]
        os.system('cp {} {}'.format(os.path.join(src_path, file), os.path.join(dst_path, file)))

if __name__ == '__main__':
    file_sample('/home/****/data/DOTA/dota_train_1024/test_split/images',
                '/home/****/data/DOTA/dota_train_1024/unlabeled_dior/images', 8000)