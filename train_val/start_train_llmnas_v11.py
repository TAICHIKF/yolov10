#!/usr/bin/env python3

'''
yolo detect train data=coco.yaml model=yolov10n/s/m/b/l/x.yaml epochs=500 batch=256 imgsz=640 device=0,1,2,3,4,5,6,7
yolo detect train data=coco.yaml model=yolov10m.yaml epochs=100 batch=16 imgsz=640 device=0,1,2,3'
yolo detect train data=coco.yaml model=yolov10m.yaml epochs=100 batch=16 imgsz=640 device=0 resume model=r'D:\code\yolov10\runs\detect\train\weights\last.pt'
ps aux | grep anaconda3/envs/yolo | grep -v grep | awk '{print $2}' | xargs kill -9
'''

import os
import yaml
import json
from datetime import datetime
from ultralytics import YOLO
# from LLM.llm_generate import generate_new_structure_using_llm
from LLM.llm_generate_module import generate_new_structure_using_llm
from LLM.llm_generate_module_v11 import generate_new_structure_using_llm_v11
from data_utils.compute_nas_score import compute_nas_score_yolov8
from data_utils.data_process import num_percent, save_model_info, get_max, clear_gpu_memory
from data_utils.extract_scales import extract_parameters
#-----------------------------------------------------------------

yolo_model = 1  # True, 是否训练baseline模型（v8 & v11）

version = 'v11'  # v8, v11
Train_flag = 0 # 如果测试llm生成架构时，值为0，训练时为1
scale = 'n' # n s m l x
more_modules = 1 # llm生成架构时，更改模块类型则为1

total_iterations = 20 # 假设循环5次

percent = '100'
api_type = 'Poe'  # 设置API类型，可以是 'Poe' 或其他: qwen

task_name_template = f'yolo{version}plus'  # 任务名称的模板
coco_data = './train_val/cfg_llm/data/coco.yaml'
coco_dir = '/xmnt/mnt_nfs_qynas_v4/kongfei/data/coco' # a04 - u404


# Define the scales dictionary
YOLOv11_scales = {
    "n": [0.50, 0.25, 1024], # summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
    "s": [0.50, 0.50, 1024], # summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
    "m": [0.50, 1.00, 512], # summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
    "l": [1.00, 1.00, 512], # summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
    "x": [1.00, 1.50, 512] # summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs
}

YOLOv8_scales = {
    "n": [0.33, 0.25, 1024],  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
    "s": [0.33, 0.50, 1024],  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
    "m": [0.67, 0.75, 768],   # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
    "l": [1.00, 1.00, 512],   # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
    "x": [1.00, 1.25, 512],   # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
}

# Check which scale dictionary to use
if version == "v8":
    scales_dict = YOLOv8_scales
elif version == "v11":
    scales_dict = YOLOv11_scales
else:
    print(f"Invalid version '{version}'. 'v8' or 'v11'.")
    exit(1)

if scale in scales_dict:
    value = scales_dict[scale]
    params = extract_parameters(scale, version)
    # print(f"Scale '{scale}' details:")
    # print(f"  Depth: {value[0]}, Width: {value[1]}, Max Channels: {value[2]}")
    print(f"YOLO{version}{scale} Layers: {params['layers']}, Parameters: {params['parameters']}, Gradients: {params['gradients']}, GFLOPs: {params['GFLOPs']}")
else:
    print(f"Invalid key '{scale}'. Please enter one of: {', '.join(scales_dict.keys())}")


#-----------------------------------------------------------------
#-----------------------------------------------------------------

# 根据条件修改配置
if percent=='100':
    # 加载yaml文件
    with open(coco_data, 'r') as file:
        config = yaml.safe_load(file)
    config['train'] = f'train2017.txt'
else:
    # 加载yaml文件
    with open(coco_data, 'r') as file:
        config = yaml.safe_load(file)
    num_percent(percent, coco_dir)
    config['train'] = f'train2017_{percent}percent.txt'
# 保存配置
with open(coco_data, 'w') as file:
    yaml.safe_dump(config, file)


if yolo_model:
    task_name = f'yolo{version}n'
    model = YOLO(f'{task_name}.yaml', verbose=True)
    model.train(data=coco_data, epochs=520, imgsz=640, batch=192, device=[0,1], name=task_name, cache=True, plots=True, pretrained=True,
                resume=True, model='/home/kongfei/code/yolov10/runs/detect/yolov11n/weights/last.pt'
                )
    save_dir=fr'./runs/detect/{task_name}'
    get_max(fr'{save_dir}2/results.csv')
    print(f"训练完成: {task_name}")
    del model  # Delete model instance after each iteration
    clear_gpu_memory()  # Clear memory

else: 
    # 获取今天的日期，格式化为 'YYYYMMDD'
    today_time = datetime.now().strftime("%Y%m%d")
    # 循环生成不同的任务名称
    max_score = 0
    max_score_list = []
    best_task_name_list = []
    best_new_yaml_list = []
    best_new_yaml_layers_list = []
    best_new_yaml_parameters_list = []
    best_new_yaml_gflops_list = []
    
    if more_modules:
        dir_path = f"./train_val/cfg_llm/models_new/{version}_{api_type}_{total_iterations}_{scale}"
    else:
        dir_path = f"./train_val/cfg_llm/models/{version}_{api_type}_{total_iterations}_{scale}"
        
    # 确保文件所在的目录存在，如果不存在则创建
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"目录已创建: {dir_path}")
    
    # 文件路径，用于存储 task_name 和 zen_score
    score_file = f"{dir_path}/task_scores.json"
    # 初始化字典
    score_dict = {}
    # 检查是否有已有的记录
    if os.path.exists(score_file):
        with open(score_file, "r") as f:
            score_dict = json.load(f)


    for i in range(1, total_iterations + 1):
        # 动态生成 task_name 和文件路径
        task_name = f'{task_name_template}{i}'  # 生成 task_name：yolov8puls1, yolov8puls2, ...
        # 动态生成文件路径，包括今天的日期
        file_path = os.path.join(dir_path, f"{task_name}.yaml")
        # 检查文件是否已经存在
        if os.path.exists(file_path):
            print(f"文件已存在，跳过: {file_path}")
        else:
            try:
                # 调用 LLM API 生成新的网络结构
                if version == 'v8':
                    new_structure = generate_new_structure_using_llm(scale, api_type, max_score_list, best_new_yaml_list, best_new_yaml_layers_list, best_new_yaml_parameters_list, best_new_yaml_gflops_list, params, more_modules)
                elif version == 'v11':
                    new_structure = generate_new_structure_using_llm_v11(scale, api_type, max_score_list, best_new_yaml_list, best_new_yaml_layers_list, best_new_yaml_parameters_list, best_new_yaml_gflops_list, params, more_modules)
                    
                # print(f"生成的新结构: {new_structure}")
                # 将新结构写入 YAML 文件
                with open(file_path, "w") as file:
                    file.write(new_structure)
                print(f"YAML 文件已保存到: {file_path}")
                # 使用生成的 YAML 文件进行模型训练
                print("# file_path:", file_path)
                new_model = YOLO(file_path, verbose=True)
            
            except Exception as e:
                print(f"生成时发生错误: {e}")
                print("重新生成网络结构...")
                # 如果报错，重新生成结构
                if version == 'v8':
                    new_structure = generate_new_structure_using_llm(scale, api_type, max_score_list, best_new_yaml_list, best_new_yaml_layers_list, best_new_yaml_parameters_list, best_new_yaml_gflops_list, params, more_modules)
                elif version == 'v11':
                    new_structure = generate_new_structure_using_llm_v11(scale, api_type, max_score_list, best_new_yaml_list, best_new_yaml_layers_list, best_new_yaml_parameters_list, best_new_yaml_gflops_list, params, more_modules)
                    
                # 将新结构写入 YAML 文件
                with open(file_path, "w") as file:
                    file.write(new_structure)
                print(f"YAML 文件已保存到: {file_path}")
                new_model = YOLO(file_path, verbose=True)
                
            summary_info = new_model.info(detailed=False, verbose=True)
            # 搜索用做计算分数的gpu id
            info = compute_nas_score_yolov8(gpu=4, model=new_model.model.cuda(4))   
            zen_score = round(float(info['avg_nas_score']), 4)
            new_yaml_content, layers, parameters, gflops = save_model_info(task_name, file_path, summary_info, zen_score, version,  scale)
            print(f"# max_score_list: {max_score_list}")
            print(f"# layers_list: {best_new_yaml_layers_list}")
            print(f"# parameters_list: {best_new_yaml_parameters_list}")
            print(f"# gflops_list: {best_new_yaml_gflops_list}")
            print(f"# {task_name}--{layers} layers, {zen_score} score, {parameters} parameters, {gflops} gflops")

            # 判断生成的架构是否符合约束条件
            if zen_score > max_score and layers < params['layers']*2 and parameters < params['parameters'] and gflops < params['GFLOPs']:
                # 保存 task_name 和 zen_score 到字典
                score_dict[task_name] = zen_score
                # 将字典保存到文件
                with open(score_file, "w") as f: 
                    json.dump(score_dict, f)
                    
                print("################  Condition met!  ################")
                print(f"# layers: {layers}, max layers: {params['layers'] * 2}")
                print(f"# zen_score: {zen_score}, max_score: {max_score}")
                print(f"# parameters: {parameters}, params['parameters']: {params['parameters']}")
                print(f"# gflops: {gflops}, params['GFLOPs']: {params['GFLOPs']}")
                
                max_score = zen_score
                max_score_list.append(max_score)
                best_task_name_list.append(task_name)
                best_new_yaml_list.append(new_yaml_content)
                best_new_yaml_layers_list.append(layers)
                best_new_yaml_parameters_list.append(parameters)
                best_new_yaml_gflops_list.append(gflops)
                    
            # 添加到 best_arch_list 和 max_score_list 时，同时检查是否已经有 3 个元素
            if len(max_score_list) > 5:
                max_score_list.pop(0)  # 删除最前面的元素
                best_task_name_list.pop(0)
                best_new_yaml_list.pop(0)  # 删除最前面的元素
                best_new_yaml_layers_list.pop(0)  # 删除最前面的元素
                best_new_yaml_parameters_list.pop(0)  # 删除最前面的元素
                best_new_yaml_gflops_list.pop(0)  # 删除最前面的元素
            
            del new_model  # Delete model instance after each iteration
            clear_gpu_memory()  # Clear memory

    
    # 从文件中读取字典并查找最大得分的 task_name
    if os.path.exists(score_file):
        with open(score_file, "r") as f:
            score_dict = json.load(f)
        best_task_name = max(score_dict, key=score_dict.get)
        print(f"最大得分的任务: {best_task_name}, 分数: {score_dict[best_task_name]}")        
                
    # print("# best_task_name_list:", best_task_name_list)        
    # best_task_name = best_task_name_list[-1]
    train_task_name = f'{api_type}_{total_iterations}_{scale}_{best_task_name}'
    if version == 'v8':
        save_dir=fr'./llmnas_results/yolov8/{train_task_name}{scale}'
        project_name = 'llmnas_yolov8' 
    else:
        save_dir=fr'./llmnas_results/yolov11/{train_task_name}{scale}'
        project_name = 'llmnas_yolov11' 
    
                    
    ###############  指定配置文件，不选择最大得分的配置文件  ################            
    # best_task_name = 'yolov8plus10'       
    ########################## 手动选择配置文件  ########################
    
    if os.path.exists(fr'{save_dir}/results.png'):
        print(f"已经训练过，跳过操作: {best_task_name}")
    elif not Train_flag:
         print(f"测试架构生成，暂不执行训练！")
    else:              
        # 手动选择一个配置文件
        best_file_path = os.path.join(dir_path, f"{best_task_name}{scale}.yaml")    
        # best_file_path = os.path.join("./train_val/cfg_llm/models_new/v8_Poe_30_m/yolov8plus10.yaml")    
        print("# best_file_path:", best_file_path)    
        model = YOLO(best_file_path, verbose=False)
        model.train(data=coco_data, epochs=1000, imgsz=640, batch=128, device=[0,1], project=project_name, name=f'{train_task_name}{scale}', cache=True, plots=True, pretrained=False,
                    # resume=True, model='/home/kongfei/code/yolov10/llmnas_results/yolov8/Poe_20_n_yolov8plus17n2/weights/last.pt'
                    )
        get_max(fr'{save_dir}3/results.csv')
        print(f"训练完成: {best_task_name}{scale}.yaml")
