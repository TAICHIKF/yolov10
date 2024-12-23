


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