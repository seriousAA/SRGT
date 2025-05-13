import os
from PIL import Image
import sys
import json

def image_converter(dataset_dir, output_dir, ext):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    files = []
    image_list = os.listdir(dataset_dir)
    files = [os.path.join(dataset_dir, _) for _ in image_list]
    len_files = len(files)
    for index, img in enumerate(files):
        try:
            sys.stdout.write('\r>>Converting image %d/%d ' % (index, len_files))
            sys.stdout.flush()
            dst_img = os.path.join(output_dir, os.path.splitext(img.split('/')[-1])[0] + '.' +ext)
            if os.path.exists(dst_img):
                continue
            # if os.path.exists(os.path.join(output_dir, (img.split('/')[-1]).split('.')[0] + '.' +ext)):
            #     os.remove(os.path.join(output_dir, (img.split('/')[-1]).split('.')[0] + '.' +ext))
            im = Image.open(img)
            im.save(dst_img)
        except IOError as e:
            print('could not read:', img)
            print('error:', e)
            print('skip it\n')

    sys.stdout.write('Convert Over!\n')
    sys.stdout.flush()

def image_resize_converter(dataset_dir, output_dir, ext, resize):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    files = []
    image_list = os.listdir(dataset_dir)
    files = [os.path.join(dataset_dir, _) for _ in image_list]
    len_files = len(files)
    for index, img in enumerate(files):
        try:
            sys.stdout.write('\r>>Converting image %d/%d ' % (index, len_files))
            sys.stdout.flush()
            dst_img = os.path.join(output_dir, os.path.splitext(img.split('/')[-1])[0] + '.' +ext)
            if os.path.exists(dst_img):
                continue
            # if os.path.exists(os.path.join(output_dir, (img.split('/')[-1]).split('.')[0] + '.' +ext)):
            #     os.remove(os.path.join(output_dir, (img.split('/')[-1]).split('.')[0] + '.' +ext))
            os.system("convert -resize {} {} {}".format(resize, img, dst_img))
        except IOError as e:
            print('could not read:', img)
            print('error:', e)
            print('skip it\n')

    sys.stdout.write('Convert Over!\n')
    sys.stdout.flush()

def image_json_converter(src_json, dst_json):
    data = json.load(open(src_json, 'r'))
    for img in data['images']:
        img['width'] = 1024
        img['height'] = 1024
    with open(dst_json, 'w') as f_out:
        json.dump(data, f_out)
if __name__ == "__main__":
    data_dir = '/home/****/data/DOTA/dota_train_1024/unlabeled_dior/images'
    output_dir = '/home/****/data/DOTA/dota_train_1024/unlabeled_dior/images_png'
    src_json = '/home/****/data/DOTA/dota_train_1024/unlabeled_dior/instances_unlabeled_dior.json'

    # image_resize_converter(data_dir, output_dir, 'png', '1024x1024')
    image_json_converter(src_json, src_json)