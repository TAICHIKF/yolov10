#!/usr/bin/env python3

'''
yolo detect train data=coco.yaml model=yolov10n/s/m/b/l/x.yaml epochs=500 batch=256 imgsz=640 device=0,1,2,3,4,5,6,7
yolo detect train data=coco.yaml model=yolov10m.yaml epochs=100 batch=16 imgsz=640 device=0,1,2,3'
ps aux | grep anaconda3/envs/yolo | grep -v grep | awk '{print $2}' | xargs kill -9
yolo detect train data=coco.yaml model=yolov10m.yaml epochs=100 batch=16 imgsz=640 device=0 resume model=r'D:\code\yolov10\runs\detect\train\weights\last.pt'
'''

import os
import yaml
import json
import shutil
from datetime import datetime
from ultralytics import YOLO
from LLM.llm_generate import generate_new_structure_using_llm
from data_utils.compute_nas_score import compute_nas_score_yolov8
from data_utils.data_process import num_percent, save_model_info, get_max, clear_gpu_memory

#-----------------------------------------------------------------

yolov8_model = 0 # True
Train_flag = 1 # 如果测试llm生成架构时，值为0，训练时为1
scale = 's' # n s m l x

percent = '100'
api_type = 'Poe'  # 设置API类型，可以是 'Poe' 或其他: qwen

total_iterations = 11  # 假设我们循环5次
task_name_template = 'yolov8plus'  # 任务名称的模板
coco_data = './train_val/cfg_llm/data/coco.yaml'
coco_dir = '/xmnt/mnt_nfs_qynas_v4/kongfei/data/coco' # a04 - u404

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


if yolov8_model:
    task_name = 'yolov8n'
    model = YOLO(f'{task_name}.yaml')
    model.train(data=coco_data, epochs=2, imgsz=640, batch=128, device=[0,2], name=task_name, cache=True, plots=True)
    # model.train(data='coco8.yaml', epochs=100, imgsz=640, device=[4,], name='train_v11n', cache=True, plots=True, resume=True, model='/home/kongfei/code/yolov10/runs/detect/train_10n/weights/last.pt')
    save_dir=fr'.\runs\detect\{task_name}'
    get_max(fr'{save_dir}\results.csv')
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
    best_new_yaml_parameters_list = []
    
    dir_path = f"./train_val/cfg_llm/model/{api_type}_{total_iterations}"
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
                new_structure = generate_new_structure_using_llm(api_type, max_score_list, best_new_yaml_list, best_new_yaml_parameters_list)
                # print(f"生成的新结构: {new_structure}")
                # 将新结构写入 YAML 文件
                with open(file_path, "w") as file:
                    file.write(new_structure)
                print(f"YAML 文件已保存到: {file_path}")
                # 使用生成的 YAML 文件进行模型训练
                new_model = YOLO(file_path, verbose=True)
            
            except Exception as e:
                print(f"生成时发生错误: {e}")
                print("重新生成网络结构...")
                # 如果报错，重新生成结构
                new_structure = generate_new_structure_using_llm(api_type, max_score_list, best_new_yaml_list, best_new_yaml_parameters_list)
                # 将新结构写入 YAML 文件
                with open(file_path, "w") as file:
                    file.write(new_structure)
                print(f"YAML 文件已保存到: {file_path}")
                new_model = YOLO(file_path, verbose=True)
                
            summary_info = new_model.info(detailed=False, verbose=True)
            info = compute_nas_score_yolov8(gpu=0, model=new_model.model.cuda(0))
            zen_score = round(float(info['avg_nas_score']), 4)
            new_yaml_content, parameters = save_model_info(task_name, file_path, summary_info, zen_score)
            print(f"# max_score_list: {max_score_list}")
            print(f"# parameters_list: {best_new_yaml_parameters_list}")
            print(f"# {task_name}--{zen_score}-{parameters} parameters")

           # 保存 task_name 和 zen_score 到字典
            score_dict[task_name] = zen_score
            # 将字典保存到文件
            with open(score_file, "w") as f:
                json.dump(score_dict, f)


            if zen_score > max_score:
                max_score = zen_score
                max_score_list.append(max_score)
                best_task_name_list.append(task_name)
                best_new_yaml_list.append(new_yaml_content)
                best_new_yaml_parameters_list.append(parameters)
                    
            # 添加到 best_arch_list 和 max_score_list 时，同时检查是否已经有 10 个元素
            if len(max_score_list) > 3:
                max_score_list.pop(0)  # 删除最前面的元素
                best_task_name_list.pop(0)
                best_new_yaml_list.pop(0)  # 删除最前面的元素
                best_new_yaml_parameters_list.pop(0)  # 删除最前面的元素
            
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
    train_task_name = f'{api_type}_{total_iterations}_{best_task_name}'
    save_dir=fr'./runs/detect/llmnas_yolov8/{train_task_name}'
    # 检查 save_dir 是否存在
    if os.path.exists(save_dir) and Train_flag:
        # 如果 save_dir 存在，并且其中没有 results.png 文件
        if not os.path.exists(os.path.join(save_dir, 'results.png')):
            # 删除现有的 save_dir 目录
            shutil.rmtree(save_dir)
            print(f"目录 {save_dir} 已删除")
            
            # 重新创建 save_dir
            os.makedirs(save_dir)
            print(f"目录 {save_dir} 已重新创建")
    else:
        # 如果 save_dir 不存在，直接创建
        os.makedirs(save_dir)
        print(f"目录 {save_dir} 已创建")
    
    if os.path.exists(fr'{save_dir}/results.png'):
        print(f"已经训练过，跳过操作: {best_task_name}")
    elif not Train_flag:
         print(f"测试架构生成，暂不执行训练！")
    else:              
        best_file_path = os.path.join(dir_path, f"{best_task_name}{scale}.yaml")    
        print("# best_file_path:", best_file_path)    
        model = YOLO(best_file_path, verbose=False)
        # model.train(data=coco_data, epochs=1000, imgsz=640, batch=256, device=[6,7], project='llmnas_yolov8', name=f'{train_task_name}{scale}', cache=True, plots=True, resume=True, model='/home/kongfei/code/yolov10/llmnas_yolov8/Poe_10_yolov8plus8/weights/last.pt')
        # model.train(data=coco_data, epochs=1000, imgsz=640, batch=512, device=[0], project='llmnas_yolov8', name=train_task_name, cache=True, plots=True)
        # model.train(data='coco8.yaml', epochs=100, imgsz=640, device=[4,], name='train_v11n', cache=True, plots=True, resume=True, model='/home/kongfei/code/yolov10/runs/detect/train_10n/weights/last.pt')

        get_max(fr'{save_dir}/results.csv')
        print(f"训练完成: {best_task_name}{scale}.yaml")





"""
训练可选参数
Argument	    Default	    Description
model	        None	    Specifies the model file for training. Accepts a path to either a .pt pretrained model or a .yaml configuration file. Essential for defining the model structure or initializing weights.
data	        None	    Path to the dataset configuration file (e.g., coco8.yaml). This file contains dataset-specific parameters, including paths to training and validation data, class names, and number of classes.
epochs	        100	        Total number of training epochs. Each epoch represents a full pass over the entire dataset. Adjusting this value can affect training duration and model performance.
time	        None	    Maximum training time in hours. If set, this overrides the epochs argument, allowing training to automatically stop after the specified duration. Useful for time-constrained training scenarios.
patience	    100	        Number of epochs to wait without improvement in validation metrics before early stopping the training. Helps prevent overfitting by stopping training when performance plateaus.
batch	        16	        Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70).
imgsz	        640	        Target image size for training. All images are resized to this dimension before being fed into the model. Affects model accuracy and computational complexity.
save	        True	    Enables saving of training checkpoints and final model weights. Useful for resuming training or model deployment.
save_period	    -1	        Frequency of saving model checkpoints, specified in epochs. A value of -1 disables this feature. Useful for saving interim models during long training sessions.
cache	        False	    Enables caching of dataset images in memory (True/ram), on disk (disk), or disables it (False). Improves training speed by reducing disk I/O at the cost of increased memory usage.
device	        None	    Specifies the computational device(s) for training: a single GPU (device=0), multiple GPUs (device=0,1), CPU (device=cpu), or MPS for Apple silicon (device=mps).
workers	        8	        Number of worker threads for data loading (per RANK if Multi-GPU training). Influences the speed of data preprocessing and feeding into the model, especially useful in multi-GPU setups.
project	        None	    Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.
name	        None	    Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outputs are stored.
exist_ok	    False	    If True, allows overwriting of an existing project/name directory. Useful for iterative experimentation without needing to manually clear previous outputs.
pretrained	    True	    Determines whether to start training from a pretrained model. Can be a boolean value or a string path to a specific model from which to load weights. Enhances training efficiency and model performance.
optimizer	    'auto'	    Choice of optimizer for training. Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto for automatic selection based on model configuration. Affects convergence speed and stability.
verbose	        False	    Enables verbose output during training, providing detailed logs and progress updates. Useful for debugging and closely monitoring the training process.
seed	        0	        Sets the random seed for training, ensuring reproducibility of results across runs with the same configurations.
deterministic	True	    Forces deterministic algorithm use, ensuring reproducibility but may affect performance and speed due to the restriction on non-deterministic algorithms.
single_cls	    False	    Treats all classes in multi-class datasets as a single class during training. Useful for binary classification tasks or when focusing on object presence rather than classification.
rect	        False	    Enables rectangular training, optimizing batch composition for minimal padding. Can improve efficiency and speed but may affect model accuracy.
cos_lr	        False	    Utilizes a cosine learning rate scheduler, adjusting the learning rate following a cosine curve over epochs. Helps in managing learning rate for better convergence.
close_mosaic	10	        Disables mosaic data augmentation in the last N epochs to stabilize training before completion. Setting to 0 disables this feature.
resume	        False	    Resumes training from the last saved checkpoint. Automatically loads model weights, optimizer state, and epoch count, continuing training seamlessly.
amp	True	    Enables     Automatic Mixed Precision (AMP) training, reducing memory usage and possibly speeding up training with minimal impact on accuracy.
fraction	    1.0	        Specifies the fraction of the dataset to use for training. Allows for training on a subset of the full dataset, useful for experiments or when resources are limited.
profile 	    False	    Enables profiling of ONNX and TensorRT speeds during training, useful for optimizing model deployment.
freeze	        None	    Freezes the first N layers of the model or specified layers by index, reducing the number of trainable parameters. Useful for fine-tuning or transfer learning.
lr0	0.01	    Initial     learning rate (i.e. SGD=1E-2, Adam=1E-3) . Adjusting this value is crucial for the optimization process, influencing how rapidly model weights are updated.
lrf	0.01	    Final       learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with schedulers to adjust the learning rate over time.
momentum	    0.937	    Momentum factor for SGD or beta1 for Adam optimizers, influencing the incorporation of past gradients in the current update.
weight_decay	0.0005	    L2 regularization term, penalizing large weights to prevent overfitting.
warmup_epochs	3.0	        Number of epochs for learning rate warmup, gradually increasing the learning rate from a low value to the initial learning rate to stabilize training early on.
warmup_momentum	0.8	        Initial momentum for warmup phase, gradually adjusting to the set momentum over the warmup period.
warmup_bias_lr	0.1	        Learning rate for bias parameters during the warmup phase, helping stabilize model training in the initial epochs.
box	            7.5	        Weight of the box loss component in the loss function, influencing how much emphasis is placed on accurately predicting bounding box coordinates.
cls	            0.5	        Weight of the classification loss in the total loss function, affecting the importance of correct class prediction relative to other components.
dfl    	        1.5   	    Weight of the distribution focal loss, used in certain YOLO versions for fine-grained classification.
pose	        12.0	    Weight of the pose loss in models trained for pose estimation, influencing the emphasis on accurately predicting pose keypoints.
kobj	        2.0	        Weight of the keypoint objectness loss in pose estimation models, balancing detection confidence with pose accuracy.
label_smoothing	0.0	        Applies label smoothing, softening hard labels to a mix of the target label and a uniform distribution over labels, can improve generalization.
nbs	            64	        Nominal batch size for normalization of loss.
overlap_mask	True	    Determines whether segmentation masks should overlap during training, applicable in instance segmentation tasks.
mask_ratio	    4	        Downsample ratio for segmentation masks, affecting the resolution of masks used during training.
dropout	        0.0 	    Dropout rate for regularization in classification tasks, preventing overfitting by randomly omitting units during training.
val	            True	    Enables validation during training, allowing for periodic evaluation of model performance on a separate dataset.
plots	        False	    Generates and saves plots of training and validation metrics, as well as prediction examples, providing visual insights into model performance and learning progression.
"""


""" 
CN

训练可选参数
参数名           默认值          描述
model            None          指定训练的模型文件。接受 .pt 预训练模型或 .yaml 配置文件的路径。对于定义模型结构或初始化权重是必需的。
data             None          数据集配置文件的路径（例如 coco8.yaml）。该文件包含数据集特定的参数，包括训练和验证数据的路径、类别名称以及类别数量。
epochs           100           总训练轮数。每一轮表示对整个数据集进行一次完整的遍历。调整此值会影响训练时间和模型性能。
time             None          最大训练时间（小时）。如果设置，该参数会覆盖 epochs 参数，允许训练在指定的持续时间后自动停止。适用于时间有限的训练场景。
patience         100           在验证指标没有改善的情况下，等待的轮数后才会提前停止训练。通过在性能停滞时停止训练来帮助防止过拟合。
batch            16            批量大小，有三种模式：设置为整数（例如，batch=16），自动模式用于 60% GPU 内存利用率（batch=-1），或指定利用率分数的自动模式（batch=0.70）。
imgsz            640           训练的目标图像大小。所有图像在输入模型之前都会调整为此尺寸。影响模型的准确性和计算复杂度。
save             True          启用保存训练检查点和最终模型权重。对于恢复训练或模型部署非常有用。
save_period      -1            保存模型检查点的频率，以轮数为单位指定。值为 -1 时禁用此功能。适用于在长时间训练过程中保存中间模型。
cache            False         启用将数据集图像缓存到内存（True/ram）、磁盘（disk）或禁用（False）。通过减少磁盘 I/O 提高训练速度，但会增加内存使用。
device           None          指定用于训练的计算设备：单个 GPU（device=0）、多个 GPU（device=0,1）、CPU（device=cpu）或 Apple Silicon 的 MPS（device=mps）。
workers          8             数据加载的工作线程数量（如果是多 GPU 训练，则按 RANK）。影响数据预处理和输入模型的速度，特别是在多 GPU 设置中非常有用。
project          None          训练输出保存的项目目录名称。允许有组织地存储不同的实验。
name             None          训练运行的名称。用于在项目文件夹中创建子目录，以存储训练日志和输出。
exist_ok         False         如果为 True，允许覆盖现有的 project/name 目录。适用于迭代实验，而无需手动清除以前的输出。
pretrained       True          决定是否从预训练模型开始训练。可以是布尔值或特定模型的字符串路径，用于加载权重。提高训练效率和模型性能。
optimizer        'auto'        选择训练的优化器。选项包括 SGD、Adam、AdamW、NAdam、RAdam、RMSProp 等，或自动选择基于模型配置的优化器。影响收敛速度和稳定性。
verbose          False         启用训练过程中的详细输出，提供详细日志和进度更新。适用于调试和密切监控训练过程。
seed             0             设置训练的随机种子，确保在相同配置下运行时结果的可重复性。
deterministic    True          强制使用确定性算法，确保可重复性，但可能会因限制非确定性算法而影响性能和速度。
single_cls        False        在多类别数据集中将所有类别视为一个单一类别进行训练。适用于二分类任务或专注于物体存在而非分类时。
rect             False         启用矩形训练，优化批量组成以减少填充。可以提高效率和速度，但可能会影响模型准确性。
cos_lr           False         使用余弦学习率调度器，在轮次中调整学习率以跟随余弦曲线。帮助管理学习率以实现更好的收敛。
close_mosaic     10            在最后 N 轮中禁用马赛克数据增强，以在训练完成前稳定训练。设置为 0 会禁用此功能。
resume           False         从最后一个保存的检查点恢复训练。自动加载模型权重、优化器状态和轮数，继续训练而不会中断。
amp              True          启用自动混合精度（AMP）训练，减少内存使用，并可能加速训练，对准确性影响最小。
fraction         1.0           指定用于训练的数据集分数。允许在完整数据集的子集上进行训练，适用于实验或资源有限的情况下。
profile          False         启用训练期间 ONNX 和 TensorRT 的性能分析，有助于优化模型部署。
freeze           None          冻结模型的前 N 层或按索引指定的层，减少可训练参数的数量。适用于微调或迁移学习。
lr0              0.01          初始学习率（例如 SGD=1E-2, Adam=1E-3）。调整此值对于优化过程至关重要，影响模型权重更新的速度。
lrf              0.01          最终学习率作为初始学习率的分数 = (lr0 * lrf)，与调度器一起使用，以调整学习率随时间变化。
momentum         0.937         SGD 的动量因子或 Adam 优化器的 beta1，影响当前更新中过去梯度的纳入。
weight_decay     0.0005        L2 正则化项，惩罚大权重以防止过拟合。
warmup_epochs    3.0           学习率预热的轮数，从低值逐渐增加到初始学习率，以稳定训练早期的过程。
warmup_momentum  0.8           预热阶段的初始动量，逐渐调整到设置的动量值。
warmup_bias_lr   0.1           预热阶段偏置参数的学习率，帮助在初期稳定模型训练。
box              7.5           损失函数中框损失组件的权重，影响对准确预测边界框坐标的重视程度。
cls              0.5           分类损失在总损失函数中的权重，影响正确类别预测的重要性。
dfl              1.5           分布焦点损失的权重，某些 YOLO 版本用于细粒度分类。
pose             12.0          姿态损失在姿态估计模型中的权重，影响对准确预测姿态关键点的重视程度。
kobj             2.0           姿态估计模型中关键点对象性损失的权重，平衡检测信心与姿态准确性。
label_smoothing   0.0          应用标签平滑，将硬标签软化为目标标签和标签均匀分布的混合，可以提高泛化能力。
nbs              64            用于标准化损失的名义批量大小。
overlap_mask     True          决定在训练期间分割掩码是否应重叠，适用于实例分割任务。
mask_ratio       4             分割掩码的下采样比率，影响训练期间使用的掩码的分辨率。
dropout          0.0           分类任务中的 dropout 率，通过在训练期间随机忽略单位来防止过拟合。
val              True          启用训练期间的验证，允许定期评估模型在单独数据集上的性能。
plots            False         生成并保存训练和验证指标的图表，以及预测示例，为模型性能和学习进展提供视觉洞察。
"""