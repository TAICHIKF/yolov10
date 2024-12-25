import os
from ultralytics import YOLO 
import time
from compute_nas_score import compute_nas_score_yolov8
from data_process import num_percent, save_model_info, get_max, clear_gpu_memory





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
    gpu=5
    zen_score_list = []
    for i in range(1, 10):
        # 动态生成 task_name 和文件路径
        task_name = f'{task_name_template}{i}'  # 生成 task_name：yolov8puls1, yolov8puls2, ...
        # 动态生成文件路径，包括今天的日期
        dir_path = f"./llmv8/Poe_25_20241223"
        file_path = os.path.join(dir_path, f"{task_name}.yaml")   
        
        task_name = 'yolov8n' # 7.844
        # task_name = 'yolov8s' #  7.967[7.894953086972237, 8.074837878346443, 8.06353472173214, 7.763306260108948, 7.746582180261612, 7.930160015821457, 7.916775241494179, 7.853617012500763, 7.992964521050453] 7.915192324254248
        # task_name = 'yolov8m' #  9.106
        # task_name = 'yolov8l' # 10.29
        # task_name = 'yolov8x' # 10.44, 10.24, 10.36
        model = YOLO(f'{task_name}.yaml')
        # model = YOLO(file_path)
        # model.info(detailed=True, verbose=True)
        start_timer = time.time()
        info = compute_nas_score_yolov8(gpu=gpu, model=model.model.cuda(gpu), repeat=64)
        time_cost = (time.time() - start_timer) / 64
        zen_score = info['avg_nas_score']
        zen_score_list.append(f'{zen_score:.4g}')
        # print(info)
        print(f'zen-score={zen_score:.4g}, time cost={time_cost:.4g} second(s)')
        
                    
        del model  # Delete model instance after each iteration
        clear_gpu_memory()  # Clear memory
    average_score = sum(zen_score_list) / len(zen_score_list)
    print(zen_score_list, average_score)