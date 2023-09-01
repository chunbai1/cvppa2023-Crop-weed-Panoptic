import os, random, shutil

random.seed(0)

source_path = '/data/PhenoBench_1793_386/val/'
target_path = '/data/PhenoBench_1793_386/train/'

name_path = '/data/PhenoBench/val/images'
name_list = os.listdir(name_path)
name_number = len(name_list)
rate = 0.5
pick_number = int(rate * name_number)
sample = random.sample(name_list, pick_number)

data_set = ['images', 'plant_instances', 'semantics']
angle_set = ['0', '90', '180', '270']

for name in sample:
    for set_name in data_set:
        for angle in angle_set:
            source_dir = os.path.join(source_path, set_name, '{}_'.format(angle)+name)
            target_dir = os.path.join(target_path, set_name, '{}_'.format(angle)+name)
            shutil.move(source_dir, target_dir)    


# def moveFile(fileDir, tarDir):
#     angle_set = [0, 90, 180, 270]
#     for angle in angle_set:

#     pathDir = os.listdir(fileDir)  # 取图片的原始路径
#     filenumber = len(pathDir)
#     rate = 0.1  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
#     picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
#     sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
#     print(sample)
#     for name in sample:
#         shutil.move(fileDir + name, tarDir + "\\" + name)
#     return

# if __name__ == '__main__':
#     fileDir = '/data/PhenoBench/val'  # 源图片文件夹路径
#     tarDir = '/data/PhenoBench_1793_386/val'  # 移动到新的文件夹路径
#     moveFile(fileDir, tarDir)
