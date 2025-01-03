import os
from ultralytics import YOLO 
import time
from compute_nas_score import compute_nas_score_yolov8, compute_nas_score_yolov8_v2
from data_process import num_percent, save_model_info, get_max, clear_gpu_memory
import numpy as np




def model_info(task_name, summary_info):
    
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
        
        
if __name__ == '__main__':
    task_name_template = 'yolov8plus'  # 任务名称的模板
    gpu=0
  
    for i in range(1, 2):
        zen_score_list = []
        for j in range(0,20):
            # 动态生成 task_name 和文件路径
            # 动态生成文件路径，包括今天的日期
            # dir_path = f"./llmv8/Poe_50_20241225"
            # task_name = f'{task_name_template}{i}'  # 生成 task_name：yolov8puls1, yolov8puls2, ...
            # file_path = os.path.join(dir_path, f"{task_name}.yaml")   
            # model = YOLO(file_path)
            
            # task_name = 'yolov8n' # yolov8n:7.854-[7.921, 7.856, 7.817, 7.768, 7.857, 7.935, 7.886, 8.008, 7.954, 7.924, 7.9, 7.745, 7.866, 8.089, 7.638, 7.68, 7.741, 7.963, 7.804, 7.728]
            task_name = 'yolov8s' # yolov8s:7.833-[7.73, 7.796, 7.788, 7.878, 7.758, 8.041, 7.888, 7.946, 7.763, 7.782, 7.774, 7.766, 7.886, 7.824, 7.888, 7.765, 7.826, 7.954, 7.738, 7.879]
            # task_name = 'yolov8m' #  yolov8m:9.114-[9.141, 9.258, 8.963, 9.078, 9.185, 9.101, 9.174, 9.044, 9.172, 9.154, 9.185, 9.115, 9.209, 9.092, 9.125, 9.135, 8.906, 9.103, 9.134, 9.003]
            # task_name = 'yolov8l' # yolov8l:10.32-[10.31, 10.39, 10.37, 10.37, 10.26, 10.36, 10.35, 10.37, 10.28, 10.28, 10.27, 10.36, 10.28, 10.26, 10.35, 10.32, 10.25, 10.32, 10.29, 10.3]
            # task_name = 'yolov8x' # yolov8x:10.35-[10.33, 10.32, 10.38, 10.3, 10.4, 10.36, 10.27, 10.3, 10.38, 10.38, 10.38, 10.37, 10.25, 10.32, 10.4, 10.31, 10.43, 10.37, 10.37, 10.33]
            model = YOLO(f'{task_name}.yaml')
            # model = YOLO('yolov8x.pt')

            # model.info(detailed=True, verbose=True)
            start_timer = time.time()
            info = compute_nas_score_yolov8_v2(gpu=gpu, model=model.model.cuda(gpu), repeat=32, verbose=False)
            time_cost = (time.time() - start_timer) / 32
            zen_score = info['avg_nas_score']
            zen_score_list.append(float(f'{zen_score:.4g}'))
            # print(info)
            # print(f'zen-score={zen_score:.4g}, time cost={time_cost:.4g} second(s)')


            del model  # Delete model instance after each iteration
            clear_gpu_memory()  # Clear memory
        # zen_score_list = [float(score) for score in zen_score_list]
        average_score = sum(zen_score_list) / len(zen_score_list)
        avg_nas_score = np.mean(zen_score_list)
        std_nas_score = np.std(zen_score_list)
        print(f'{task_name}:{average_score:.4g}-{avg_nas_score:.4g}-{std_nas_score:.4g}--{zen_score_list}')