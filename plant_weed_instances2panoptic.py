import cv2
import numpy as np
import os
import json
from PIL import Image
from pycocotools import mask
import datetime
import random
def get_random_color():
    """获取一个随机的颜色"""
    r = lambda: random.uniform(0,1)
    return [r(),r(),r(),1]

ROOT_DIR = '/data/PhenoBench_1793_386/'
OUT_DIR = '/data/PhenoBench_1793_386/annotations/'
INFO = {
    "description": "plant panoptic segmentation dataset",
    "url": "http",
    "version" : "0.1.0",
    "year": 2023,
    "contributor": "little bai",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}
LICENSES = [
    {
        "id": 1,
        "name": "license_001",
        "url" : "http://ipiu.xidian.edu.cn"
    },
    {
        "id": 2,
        "name": "license_002",
        "url" : "http://ipiu.xidian.edu.cn"
    }
]
CATEGORIES = [
    {
        'supercategory':"crop_parent", 'isthing':1, 'id':1, 'name': 'crop'
    },
    {
        'supercategory':"weed_parent", 'isthing':0, 'id':2, 'name':'weed'
    }
]

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


# create segmentation_info fo annotation
def create_segmentation_info(segmentation_id, category_id, iscrowd, bounding_box=None, area=None):
    segmentation_info = {
        "id": segmentation_id,
        "category_id":category_id,
        "iscrowd": iscrowd,
        "bbox": list(map(int, bounding_box)),
        'area': int(area)
    }
    # print(type(segmentation_info["bbox"]))
    # print(type(segmentation_info["bbox"][0]))
    # print(type(segmentation_info["area"]))
    return segmentation_info

def ins_sem_2_panoptic():
    set_names = ['train', 'val']
    for setname in set_names:
        coco_output = {
            "info" : INFO,
            "licenses": LICENSES,
            "images":[],
            "annotations":[],
            "categories":CATEGORIES
        }
        save_panoptic_dir = os.path.join(OUT_DIR, 'panoptic_plant_weed_rotate_{}'.format(setname))
        if not os.path.exists(save_panoptic_dir):
            os.makedirs(save_panoptic_dir)
        img_id = 1
        img_dir = os.path.join(ROOT_DIR, setname, 'images')
        img_name_list = sorted(os.listdir(img_dir))
        for img_name in img_name_list:
            img = Image.open(os.path.join(img_dir, img_name))
            img_info = create_image_info(img_id, img_name, img.size)
            coco_output["images"].append(img_info)

            annotation_info = {
                "segments_info": [],
                "file_name": img_name,
                "image_id": img_id
            }
            sem_dir = os.path.join(ROOT_DIR, setname, 'semantics', img_name)
            ins_dir = os.path.join(ROOT_DIR, setname, 'plant_instances', img_name)
            sem_array = np.array(cv2.imread(sem_dir, cv2.CV_16UC1))
            ins_array = np.array(cv2.imread(ins_dir, cv2.CV_16UC1))
            sem_array[sem_array == 3] = 1
            sem_array[sem_array == 4] = 2

            panoptic_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
            
            index = np.unique(ins_array)
            for idx in index:
                if idx == 0:
                    continue
                
                mask_ins = np.zeros((1024, 1024), dtype=np.uint8)
                mask_ins[ins_array == idx] = 1
                binary_mask_encoded = mask.encode(np.asfortranarray(mask_ins))
                area = mask.area(binary_mask_encoded)
                bounding_box = mask.toBbox(binary_mask_encoded)
                
                if np.unique(sem_array[mask_ins == 1]) == 1:
                    category_id = 1
                elif np.unique(sem_array[mask_ins == 1]) == 2:
                    category_id = 2
                color = list(np.random.choice(range(256), size=3))
                panoptic_img[ins_array == idx] = color
                segmentation_id = int(color[0] + color[1] * 256 + color[2] * 256 * 256)
                segmentation_info = create_segmentation_info(segmentation_id=segmentation_id, category_id=category_id , iscrowd=0, bounding_box=bounding_box, area=area)
                annotation_info["segments_info"].append(segmentation_info)

            coco_output["annotations"].append(annotation_info)
            cv2.cvtColor(panoptic_img, cv2.COLOR_RGB2BGR, panoptic_img)
            cv2.imwrite(os.path.join(save_panoptic_dir, img_name), panoptic_img)
            # panoptic_img = Image.fromarray(panoptic_img)
            # panoptic_rgb = panoptic_img.convert(mode='RGB')
            # panoptic_rgb.save(os.path.join(save_panoptic_dir, img_name), quality=95)
            
            img_id += 1
            print(img_id)
        panoptic_json = json.dumps(coco_output)
        save_json_dir = os.path.join(OUT_DIR, 'panoptic_plant_weed_rotate_{}.json'.format(setname))
        with open(save_json_dir, 'w') as f:
            f.write(panoptic_json)
        f.close()

        
if __name__=='__main__':
    ins_sem_2_panoptic()