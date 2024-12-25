import os
import json
import random
import gc
import torch
import pandas as pd
from collections import defaultdict


def clear_gpu_memory():
    torch.cuda.empty_cache()  # Release unreferenced GPU memory
    gc.collect()  # Collect garbage to release unused memory


def num_percent(percent, coco_dir):
    
    output_txt = f"{coco_dir}/train2017_{percent}percent.txt"                    # 输出的图像路径txt文件
    if os.path.exists(output_txt):
        print(f'# 已经生成了{output_txt}, 直接跳过！')
    else:
        # 设置路径
        coco_json = f"{coco_dir}/annotations_trainval2017/annotations/instances_train2017.json"  # COCO标注文件路径
        output_json = f"{coco_dir}/annotations/instances_train2017_{percent}percent.json"         # 抽取后保存的标注文件

        # 图像原路径
        # images_dir = "{coco_dir}/images/train2017"

        # 加载 COCO 标注文件
        with open(coco_json, "r") as f:
            coco_data = json.load(f)

        # 解析标注信息
        images = coco_data["images"]        # 图像信息
        annotations = coco_data["annotations"]  # 标注信息
        categories = coco_data["categories"]    # 类别信息

        # 统计每个类别的标注
        category_to_annotations = defaultdict(list)
        for ann in annotations:
            category_to_annotations[ann["category_id"]].append(ann)

        # 抽取每个类别1%的数据
        selected_annotations = []
        for cat_id, anns in category_to_annotations.items():
            num_to_select = max(1, int(len(anns) * 0.01 * int(percent)))  # 至少选取1个
            selected_annotations.extend(random.sample(anns, num_to_select))

        # 获取抽取到的图像ID，确保唯一性
        selected_image_ids = set(ann["image_id"] for ann in selected_annotations)

        # 抽取对应的图像信息
        selected_images = [img for img in images if img["id"] in selected_image_ids]

        # 保存新的标注文件
        new_coco_data = {
            "images": selected_images,
            "annotations": selected_annotations,
            "categories": categories
        }

        with open(output_json, "w") as f:
            json.dump(new_coco_data, f, indent=4)

        print(f"抽取完成！新的标注文件保存在: {output_json}")
        print(f"图像数量: {len(selected_images)}，标注数量: {len(selected_annotations)}")

        # 保存图像路径到txt文件
        with open(output_txt, "w") as f:
            for img in selected_images:
                # img_path = os.path.join(images_dir, img["file_name"])
                img_path = f"./images/train2017/{img['file_name']}"  # 相对路径
                f.write(img_path + "\n")

        print(f"图像路径已保存到: {output_txt}")

        # 统计新数据集中每个类别的标注数量
        new_category_count = defaultdict(int)
        for ann in selected_annotations:
            new_category_count[ann["category_id"]] += 1

        # 输出统计信息
        for cat in categories:
            cat_id = cat["id"]
            cat_name = cat["name"]
            print(f"{cat_name}: {new_category_count[cat_id]} 个标注")


def get_max(file_path):
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在！")
        return
    
    data = pd.read_csv(file_path)
    # 去除列名前后的空格
    data.columns = data.columns.str.strip()
    
    print("列名：", repr(data.columns))  # 显示列名

    # 获取 `metrics/mAP50-95(B)` 的最大值及对应的 epoch
    column_name = "metrics/mAP50-95(B)"  # 确保列名正确
    if column_name in data.columns:
        print("数据头部：", data.head())  # 查看前几行数据
        print("数据类型：", data[column_name].dtype)  # 查看列数据类型
        
        # 检查是否有NaN值并去除
        if data[column_name].isna().sum() > 0:
            print(f"警告：'{column_name}' 列有 {data[column_name].isna().sum()} 个缺失值，正在去除...")
            data = data.dropna(subset=[column_name])
        
        max_value = data[column_name].max()  # 最大值
        max_epoch = data.loc[data[column_name].idxmax(), "epoch"]  # 对应的 epoch
        print(f"最大 mAP50-95(B): {max_value}, 对应的 epoch: {max_epoch}")
    else:
        print(f"列 '{column_name}' 不存在，请检查列名是否正确。")


def save_model_info(task_name, file_path, summary_info):
    
    yolov8n_info = '#     YOLOv8n summary: 225 layers, 3,157,200 parameters, 3,157,184 gradients,  8.9 GFLOPs'
    # Unpack the tuple into individual variables
    layers, parameters, gradients, gflops = summary_info
    # Format the string using f-strings with comma separators and specified precision
    formatted_summary = (
        f"#{task_name} summary: {layers} layers, "
        f"{parameters:,} parameters, "
        f"{gradients:,} gradients, "
        f"{gflops:.1f} GFLOPs"
    )

    # 读取现有的 YAML 文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 检查文件末尾是否有换行符
    if not content.endswith('\n'):
        content += '\n\n' 

    # 将现有内容和新的 formatted_summary 合并
    new_content = content + yolov8n_info + '\n' + formatted_summary + '\n'

    # 将合并后的内容写回文件
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(new_content)