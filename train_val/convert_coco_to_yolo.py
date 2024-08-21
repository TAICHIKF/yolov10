

import os
import json
from pathlib import Path
from pycocotools.coco import COCO

# # COCO 数据集路径
# coco_annotation_file = r'D:\code\datasets\coco\annotations_trainval2017\annotations\instances_train2017.json'  # 训练集
# images_dir = r'D:\code\datasets\coco\images\train2017'
# labels_dir = r'D:\code\datasets\coco\labels_\train2017'

# 验证集的 COCO 注释文件路径
coco_annotation_file = r'D:\code\datasets\coco\annotations_trainval2017\annotations\instances_val2017.json'  # 验证集
images_dir = r'D:\code\datasets\coco\images\val2017'
labels_dir = r'D:\code\datasets\coco\labels_\val2017'


# 创建 labels 目录
Path(labels_dir).mkdir(parents=True, exist_ok=True)

# 加载 COCO 注释文件
coco = COCO(coco_annotation_file)

# 获取所有类别
categories = coco.loadCats(coco.getCatIds())
category_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}

# 处理每张图像
for img_id in coco.imgs:
    img_info = coco.imgs[img_id]
    img_file_name = img_info['file_name']
    img_width = img_info['width']
    img_height = img_info['height']
    
    # 获取图像中的所有标注
    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    
    label_file_name = img_file_name.replace('.jpg', '.txt')
    label_file_path = os.path.join(labels_dir, label_file_name)
    
    # 创建标签文件
    with open(label_file_path, 'w') as f:
        for ann in anns:
            # 获取类别 ID 并映射到 YOLO 类别 ID
            category_id = ann['category_id']
            yolo_class_id = category_mapping[category_id]
            
            # 获取边界框并归一化
            bbox = ann['bbox']
            x_min = bbox[0]
            y_min = bbox[1]
            bbox_width = bbox[2]
            bbox_height = bbox[3]
            
            x_center = (x_min + bbox_width / 2) / img_width
            y_center = (y_min + bbox_height / 2) / img_height
            width = bbox_width / img_width
            height = bbox_height / img_height
            
            # 写入 YOLO 格式标签
            f.write(f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print("COCO 数据集成功转换为 YOLO 格式！")
