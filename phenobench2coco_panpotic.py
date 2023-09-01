import json
from PIL import Image
import os
import numpy as np
from pycocotools import mask
import datetime

##########################################################################3
# from mode='L' format generate 'RGB' image : annotation_seg
def generate_rgb_from_L(orgin_path, save_path):
    img_name_list = sorted(os.listdir(orgin_path))

    img_path_list = [os.path.join(orgin_path, img_name) for img_name in img_name_list]
    save_path_list = [os.path.join(save_path, img_name) for img_name in img_name_list]

    for i in range(len(img_path_list)):
        img_L = Image.open(img_path_list[i])
        img_rgb = img_L.convert(mode='RGB')

        img_rgb.save(save_path_list[i], quality=95)
##########################################################################

ROOT_DIR = '/home/zhongzhou/Documents/mmlab_cuda/mmdetection/data/PhenoBench/train'
IMAGE_DIR = os.path.join(ROOT_DIR, 'images')
INSTANCES_DIR = os.path.join(ROOT_DIR, 'leaf_instances')
ANNOTATION_DIR = os.path.join(ROOT_DIR, 'leaf_annoations')

SAVE_JSON_DIR = '/home/zhongzhou/Documents/mmlab_cuda/mmdetection/data/PhenoBench_coco/annotations'
SAVE_JOSN_NAME = 'panoptic_leaf_train.json'

# create img info for annotations
def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info


## masks_generator accroding every instance for image
def masks_generator(img_name_list, mask_name_list):
    # check the data integrity
    for img_name in img_name_list:
        assert img_name in mask_name_list, "exist some img is not in the mask list"

    for mask_name in mask_name_list:
        idx = 0
        all_instances_image = np.array(Image.open(os.path.join(INSTANCES_DIR, mask_name)))
        height, width = all_instances_image.shape[:2]
        # find the unique elements of an array, return the sorted elements of an array.
        index_array = np.unique(all_instances_image)
        # every instance: 0:background, other:thing
        for index in index_array:
            if index == 0:
                continue

            instance_id = index
            instance_class = 'leaf'

            instance_mask = np.zeros((height, width), dtype=np.uint8)

            mask = all_instances_image == instance_id

            instance_mask[mask] = 255

            # 1.path
            mask_save_path = os.path.join(ANNOTATION_DIR, mask_name.split('.')[0])
            # 2.name
            instance_name =  instance_class + '_' + str(idx) + '.png'
            # save
            if not os.path.exists(mask_save_path):
                os.mkdir(mask_save_path)
            instance_mask = Image.fromarray(instance_mask)
            instance_mask.save(os.path.join(mask_save_path, instance_name), mode='L')
            ########
            idx += 1
 

# create segmentation_info fo annotation
def create_segmentation_info(segmentation_id, category_id, iscrowd, bounding_box=None, area=None):
    segmentation_info = {
        "id": segmentation_id,
        "category_id":category_id,
        "is_crowd": iscrowd,
        "bbox": list(map(int, bounding_box)),
        'area': int(area)
    }
    # print(type(segmentation_info["bbox"]))
    # print(type(segmentation_info["bbox"][0]))
    # print(type(segmentation_info["area"]))
    return segmentation_info


INFO = {
    "description": "leaf panoptic segmentation dataset",
    "url": "http",
    "version" : "0.1.0",
    "year": 2023,
    "contributor": "zhongzhou zhou",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "license_001",
        "url" : "http://zhongzhou.xidian.edu.cn"
    },
    {
        "id": 2,
        "name": "license_002",
        "url" : "http://zhongzhou.xidian.edu.cn"
    }
]
CATEGORIES = [
    {
        'supercategory':"object", 'isthing':1, 'id':1, 'name': 'leaf'
    },
    {
        'supercategory':"background", 'isthing':0, 'id':2, 'name':'soil'
    }
]


# generate annotation panoptic_leaf_train.json and panoptic_leaf_val.json
def generate_panoptic_anno_json():
    
    coco_output = {
        "info" : INFO,
        "licenses": LICENSES,
        "images":[],
        "annotations":[],
        "categories":CATEGORIES
    }

    # image id 
    image_id = 1
    # instance id 
    segmentation_id = 1

    # push all image, in the img path. sort all img.....according the img name take the img and mask.
    img_name_list = sorted(os.listdir(IMAGE_DIR))

    # go through each image
    for img_name in img_name_list:
        img_path = os.path.join(IMAGE_DIR, img_name)
        #########################################################
        img = Image.open(img_path)
        
        img_info = create_image_info(image_id, img_name, img.size)
        coco_output["images"].append(img_info)
        #########################################################\
        annotation_info = {
            "segments_info" : [],
            "file_name": img_name,
            "image_id": image_id
        }

        instace_path = os.path.join(ANNOTATION_DIR, img_name.split('.')[0])
        instance_names_list = sorted(os.listdir(instace_path))

        areas = 0
        for instance_name in instance_names_list:
            instance = Image.open(os.path.join(instace_path, instance_name))
            instance = np.array(instance, dtype=np.uint8)

            binary_mask_encoded = mask.encode(np.asfortranarray(instance))
            area = mask.area(binary_mask_encoded)
            bounding_box = mask.toBbox(binary_mask_encoded)
            category_id = 1
            
            segmentation_info = create_segmentation_info(segmentation_id=segmentation_id, category_id=category_id , iscrowd=0, bounding_box=bounding_box, area=area)
            annotation_info["segments_info"].append(segmentation_info)
            
            segmentation_id += 1
            areas += area

        # generate the last background segmentation_info 
        binary_mask_encoded = mask.encode(np.asfortranarray(instance))
        area = 1024*1024 - areas
        bounding_box = [0, 0, 1024, 1024]
        category_id = 2
        segmentation_info = create_segmentation_info(segmentation_id=segmentation_id, category_id=category_id , iscrowd=0, bounding_box=bounding_box, area=area)
        annotation_info["segments_info"].append(segmentation_info)
        segmentation_id += 1

        coco_output["annotations"].append(annotation_info)
        image_id += 1
    # save coco_output
    panopatic_json = json.dumps(coco_output)
    with open(os.path.join(SAVE_JSON_DIR, SAVE_JOSN_NAME), 'w') as f:
        f.write(panopatic_json)
    f.close()
    # return coco_output


if __name__=='__main__':
    ## L -> rgb
    # generate_rgb_from_L(orgin_path="data/PhenoBench/train/leaf_instances", save_path="data/PhenoBench_coco/annotations/panoptic_leaf_train")
    # generate_rgb_from_L(orgin_path="data/PhenoBench/val/leaf_instances", save_path="data/PhenoBench_coco/annotations/panoptic_leaf_val")
    # create annotation mask
    # img_name_list = sorted(os.listdir(IMAGE_DIR))
    # mask_name_lsit = sorted(os.listdir(INSTANCES_DIR))
    # masks_generator(img_name_list, mask_name_lsit)
    generate_panoptic_anno_json()




