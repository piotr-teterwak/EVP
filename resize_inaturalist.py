import glob
import os

import cv2
import numpy as np

from multiprocessing.pool import Pool 

from tqdm import *

IMG_DIR='data/iNaturalist-2021/2021_train/'
DST_DIR='/fsx/pteterwak/data/iNaturalist-2021-resized-164/2021_train'

in_files = glob.glob(IMG_DIR+"*/*.jpg")

def read_images():
    for img in in_files:
        image = cv2.imread(img)
        (h, w) = image.shape[:2]
        aspect_ratio = h/w
        if h < w:
            resized_image_size = (int(164/aspect_ratio) , 164)
        else:
            resized_image_size = (164, int(164 * aspect_ratio))
        resized_img = cv2.resize(image, resized_image_size)

        yield (img, image, resized_img)

resized_imgs_iter = iter(read_images())


def write_image(img_data):
    img, _, resized_img = img_data

    dst_dir = os.path.join(DST_DIR,img.split('/')[-2])
    if not os.path.exists(dst_dir):
        try:
            os.makedirs(dst_dir)
        except:
            pass
    dst_path = os.path.join(DST_DIR, '/'.join(img.split('/')[-2:]))
    cv2.imwrite(dst_path, resized_img)
    return dst_path

num_images = len(in_files)
print(num_images)


with Pool(96) as pool:
    with tqdm(total=num_images) as pbar:
        for result in pool.imap_unordered(write_image, resized_imgs_iter):
            pbar.update()
    pool.close()
    pool.join()
