import json
import random
import os
from collections import defaultdict


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


if __name__ == '__main__':
    num_percent(1)



'''
2024.12.19
抽取完成！新的标注文件保存在: {coco_dir}/annotations/instances_train2017_5percent.json
图像数量: 31960，标注数量: 42958
图像路径已保存到: {coco_dir}/train2017_5percent.txt
person: 13123 个标注
bicycle: 355 个标注
car: 2193 个标注
motorcycle: 436 个标注
airplane: 256 个标注
bus: 303 个标注
train: 228 个标注
truck: 498 个标注
boat: 537 个标注
traffic light: 644 个标注
fire hydrant: 93 个标注
stop sign: 99 个标注
parking meter: 64 个标注
bench: 491 个标注
bird: 540 个标注
cat: 238 个标注
dog: 275 个标注
horse: 329 个标注
sheep: 475 个标注
cow: 407 个标注
elephant: 275 个标注
bear: 64 个标注
zebra: 265 个标注
giraffe: 256 个标注
backpack: 436 个标注
umbrella: 571 个标注
handbag: 617 个标注
tie: 324 个标注
suitcase: 309 个标注
frisbee: 134 个标注
skis: 332 个标注
snowboard: 134 个标注
sports ball: 317 个标注
kite: 453 个标注
baseball bat: 163 个标注
baseball glove: 187 个标注
skateboard: 277 个标注
surfboard: 306 个标注
tennis racket: 240 个标注
bottle: 1217 个标注
wine glass: 395 个标注
cup: 1032 个标注
fork: 273 个标注
knife: 388 个标注
spoon: 308 个标注
bowl: 717 个标注
banana: 472 个标注
apple: 292 个标注
sandwich: 218 个标注
orange: 319 个标注
broccoli: 365 个标注
carrot: 392 个标注
hot dog: 145 个标注
pizza: 291 个标注
donut: 358 个标注
cake: 317 个标注
chair: 1924 个标注
couch: 288 个标注
potted plant: 432 个标注
bed: 209 个标注
dining table: 785 个标注
toilet: 207 个标注
tv: 290 个标注
laptop: 248 个标注
mouse: 113 个标注
remote: 285 个标注
keyboard: 142 个标注
cell phone: 321 个标注
microwave: 83 个标注
oven: 166 个标注
toaster: 11 个标注
sink: 280 个标注
refrigerator: 131 个标注
book: 1235 个标注
clock: 316 个标注
vase: 330 个标注
scissors: 74 个标注
teddy bear: 239 个标注
hair drier: 9 个标注
toothbrush: 97 个标注
'''



'''
2024.12.18
抽取完成！新的标注文件保存在: {coco_dir}/annotations/instances_train2017_1percent.json
图像数量: 7971，标注数量: 8559
图像路径已保存到: {coco_dir}/train2017_1percent.txt
person: 2624 个标注
bicycle: 71 个标注
car: 438 个标注
motorcycle: 87 个标注
airplane: 51 个标注
bus: 60 个标注
train: 45 个标注
truck: 99 个标注
boat: 107 个标注
traffic light: 128 个标注
fire hydrant: 18 个标注
stop sign: 19 个标注
parking meter: 12 个标注
bench: 98 个标注
bird: 108 个标注
cat: 47 个标注
dog: 55 个标注
horse: 65 个标注
sheep: 95 个标注
cow: 81 个标注
elephant: 55 个标注
bear: 12 个标注
zebra: 53 个标注
giraffe: 51 个标注
backpack: 87 个标注
umbrella: 114 个标注
handbag: 123 个标注
tie: 64 个标注
suitcase: 61 个标注
frisbee: 26 个标注
skis: 66 个标注
snowboard: 26 个标注
sports ball: 63 个标注
kite: 90 个标注
baseball bat: 32 个标注
baseball glove: 37 个标注
skateboard: 55 个标注
surfboard: 61 个标注
tennis racket: 48 个标注
bottle: 243 个标注
wine glass: 79 个标注
cup: 206 个标注
fork: 54 个标注
knife: 77 个标注
spoon: 61 个标注
bowl: 143 个标注
banana: 94 个标注
apple: 58 个标注
sandwich: 43 个标注
orange: 63 个标注
broccoli: 73 个标注
carrot: 78 个标注
hot dog: 29 个标注
pizza: 58 个标注
donut: 71 个标注
cake: 63 个标注
chair: 384 个标注
couch: 57 个标注
potted plant: 86 个标注
bed: 41 个标注
dining table: 157 个标注
toilet: 41 个标注
tv: 58 个标注
laptop: 49 个标注
mouse: 22 个标注
remote: 57 个标注
keyboard: 28 个标注
cell phone: 64 个标注
microwave: 16 个标注
oven: 33 个标注
toaster: 2 个标注
sink: 56 个标注
refrigerator: 26 个标注
book: 247 个标注
clock: 63 个标注
vase: 66 个标注
scissors: 14 个标注
teddy bear: 47 个标注
hair drier: 1 个标注
toothbrush: 19 个标注
'''