# import cv2
# import numpy as np

# from PIL import Image

# img = cv2.imread('/data/PhenoBench/val/annotations/05-15_00053_P0030943_plant_1889.png', cv2.CV_16UC1)
# img_array = np.array(img)
# print(np.unique(img_array))

# img = Image.open('/data/PhenoBench/val/annotations/05-15_00053_P0030943_plant_1889.png')
# img_array = np.array(img)
# print(np.unique(img_array))

import json
import os
import cv2
ROOT_DIR = '/data/PhenoBench/val'

with open('/data/PhenoBench/plant_instances_train.json') as f:
    info = json.load(f)
# for img_anno in info['annotations']:
#     img_anno['iscrowd'] = 0
# with open('{}/leaf_0.json'.format(ROOT_DIR), 'w') as output_json_file:
#     json.dump(info, output_json_file)
# print('write ok!')
num_list = []
print(info.keys())
for anno in info['annotations']:
    polygon = anno['segmentation']
    polygon_len = len(polygon)
    num_list.append(polygon_len)
print(set(num_list))
print(info['categories'])
print(info['annotations'][0])
# print(info['annotations'][133]['segmentation'])
# print(info['images'][133]['file_name'])
print(info['images'][0])
# print(info['categories'])
# img_path = os.path.join('/data/PhenoBench/val/images/', info['images'][0]['file_name'])
# bbox = info['annotations'][2]['bbox']
# img = cv2.imread(img_path)
# cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 0, 255), 2)
# cv2.imwrite('test.png', img)


# with open('/data/PhenoBench/val/plant.json') as f:
#     info = json.load(f)
# # for img_anno in info['annotations']:
# #     img_anno['iscrowd'] = 0
# # with open('{}/leaf_0.json'.format(ROOT_DIR), 'w') as output_json_file:
# #     json.dump(info, output_json_file)
# # print('write ok!')
# num_list = []
# print(info.keys())
# for anno in info['annotations']:
#     polygon = anno['segmentation']
#     polygon_len = len(polygon)
#     num_list.append(polygon_len)
# print(set(num_list))

# # print(info['annotations'][133]['segmentation'])
# print(info['images'][133]['file_name'])
