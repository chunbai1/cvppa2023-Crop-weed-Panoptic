import cv2
import numpy as np
import os, glob
import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
 
ROOT_DIR = '/data/PhenoBench/train'
IMAGE_DIR = os.path.join(ROOT_DIR, "images") 
INSTANCE_DIR = os.path.join(ROOT_DIR, "plant_instances")
ANNOTATION_DIR = os.path.join(ROOT_DIR, "plant_annotations") 
 
INFO = {
    "description": "PhenoBench_Instance plant Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": "2023",
    "contributor": "Bai",
    "date_created": "2023"
}
 
LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

# repair
CATEGORIES = [
    {
        'id': 1,
        'name': 'plant',
        'supercategory': 'none',
    }
]
  
def masks_generator(images):
    idx = 0
    for pic_name in images:
        annotation_pic = cv2.imread(os.path.join(INSTANCE_DIR, pic_name), cv2.CV_16UC1)
        h, w = annotation_pic.shape[:2]
        ids = np.unique(annotation_pic)
        for id in ids:
            if id == 0:
                continue
            instance_id = id
            class_id = 1
            instance_class = 'plant'
            # print(instance_id)
            instance_mask = np.zeros((h, w, 3),dtype=np.uint8)
            mask = annotation_pic == instance_id
            instance_mask[mask] = 255
            # print(instance_mask.max())
            mask_name = pic_name.split('.')[0] + '_' + instance_class + '_' + str(idx) + '.png'
            cv2.imwrite(os.path.join(ANNOTATION_DIR, mask_name), instance_mask)
            idx += 1
 
 
def filter_for_instances(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [f for f in files if re.match(file_types, f)]
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    # files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]
    return files
 
 
def main():
    # for root, _, files in os.walk(ANNOTATION_DIR):
    image_files = os.listdir(IMAGE_DIR)
    # masks_generator(image_files)
    # print('masks ok!')
 
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    image_id = 1
    segmentation_id = 1
    
    instance_files = os.listdir(ANNOTATION_DIR)
 
    # go through each image
    for image_filename in image_files:
        image_path = os.path.join(IMAGE_DIR, image_filename)
        image = Image.open(image_path)
        image_info = pycococreatortools.create_image_info(
                image_id, os.path.basename(image_filename), image.size)
        coco_output["images"].append(image_info)
 
        # filter for associated png annotations
        # for root, _, files in os.walk(INSTANCE_DIR):
        annotation_files = filter_for_instances(ANNOTATION_DIR, instance_files, image_filename)
 
        # go through each associated annotation
        for annotation_filename in annotation_files:
            annotation_path = os.path.join(ANNOTATION_DIR, annotation_filename)
            # print(annotation_path)
            class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]
 
            category_info = {'id': class_id, 'is_crowd': False}
            binary_mask = np.asarray(Image.open(annotation_path).convert('1')).astype(np.uint8)

            annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask,
                    image.size, tolerance=0)
 
            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)
 
                segmentation_id = segmentation_id + 1
 
        image_id = image_id + 1
 
    print('annotation ok!')
    with open('{}/plant_train_tr0.json'.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    print('write ok!')
 
 
if __name__ == "__main__":
    main()