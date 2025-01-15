'''
python ./train_val/data_utils/compute_nas_score.py --batch_size 16 --input_image_size 640 --repeat_times 32 --gpu 0 --mixup_gamma 0.01
'''

import sys
import torch
from torch import nn
import numpy as np
import argparse, time
from ultralytics import YOLO  # Import the YOLO model

# No need to import Detect if using class name checking
# from ultralytics.nn.modules.head import Detect

def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--input_image_size', type=int, default=640, help='Input image resolution.')
    parser.add_argument('--repeat_times', type=int, default=512)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mixup_gamma', type=float, default=1e-2)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, mean=0.0, std=0.001)  # 减小 std
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.constant_(m.bias, 0)
                m.running_mean.zero_()
                # m.running_var.fill_(0.001)  # 减小 running_var
                m.running_var.fill_(1.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # 使用 Xavier 初始化
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
    return net


def compute_nas_score_yolov8(gpu, model, mixup_gamma=0.01, resolution=640, batch_size=16, repeat=32, fp16=False):
    info = {}
    nas_score_list = []
    epsilon = 1e-8  # 一个很小的正数

    device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')
    dtype = torch.half if fp16 else torch.float32

    model.eval()  # 设置模型为评估模式

    # 在循环外初始化模型权重（或不重新初始化）
    # network_weight_gaussian_init(model)

    # 打印权重信息
    weights = model.state_dict()  # 获取权重
    # print(f"模型权重: {weights}") 


    with torch.no_grad():
        for repeat_count in range(repeat):
            # 不再重新初始化模型

            # 创建随机输入
            input1 = torch.randn([batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            input2 = torch.randn([batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            mixup_input = input1 + mixup_gamma * input2

            # 前向传播
            output1 = model(input1)
            output2 = model(mixup_input)

            # 提取特征或预测
            if isinstance(output1, torch.Tensor):
                features1 = output1
                features2 = output2
            elif isinstance(output1, (tuple, list)):
                features1 = output1[0]
                features2 = output2[0]
            elif isinstance(output1, dict):
                features1 = output1.get('pred', None)
                features2 = output2.get('pred', None)
                if features1 is None:
                    # 根据模型实际输出调整键名
                    features1 = output1.get('output', None)
                    features2 = output2.get('output', None)
            else:
                raise TypeError(f"Unexpected output type: {type(output1)}")

            if features1 is None or features2 is None:
                raise ValueError("Could not extract features from model outputs.")

            # 确保 features 是张量或张量的列表
            if isinstance(features1, torch.Tensor):
                # 直接计算 NAS 分数
                nas_diff = torch.abs(features1 - features2)
                # print(f"nas_diff mean: {nas_diff.mean().item()}, std: {nas_diff.std().item()}")
                nas_score = torch.sum(nas_diff, dim=tuple(range(1, features1.ndim)))
            elif isinstance(features1, (list, tuple)):
                # 对多个特征图进行计算
                nas_score = 0
                for feat1, feat2 in zip(features1, features2):
                    nas_diff = torch.abs(feat1 - feat2)
                    # print(f"nas_diff mean: {nas_diff.mean().item()}, std: {nas_diff.std().item()}")
                    nas_score += torch.sum(nas_diff, dim=tuple(range(1, feat1.ndim)))
            else:
                raise TypeError("Features are not tensors or lists of tensors.")

            nas_score = torch.mean(nas_score)

            # 检查 nas_score 值
            if torch.isnan(nas_score) or nas_score <= 0:
                print(f"Invalid nas_score: {nas_score.item()} at repeat {repeat_count}")
                continue  # 跳过此轮次

            # 计算 BN 层的缩放因子
            log_bn_scaling_factor = 0.0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var + epsilon))
                    if torch.isnan(bn_scaling_factor) or bn_scaling_factor <= 0:
                        print(f"Invalid bn_scaling_factor: {bn_scaling_factor.item()}")
                        continue  # 跳过此层
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)

            # 加一个小的 epsilon，防止对零取对数
            nas_score += epsilon
            # nas_score = torch.log(nas_score) + log_bn_scaling_factor
            # nas_score_list.append(float(nas_score))
            
            # # 计算 nas_score，取绝对值
            nas_score = torch.abs(torch.log(nas_score + epsilon)) + torch.abs(log_bn_scaling_factor)
            nas_score_list.append(float(nas_score))
            
    # 如果 nas_score_list 全部为 NaN，抛出错误
    if len(nas_score_list) == 0:
        raise ValueError("No valid nas_score values computed.")

    # 计算统计数据
    avg_nas_score = np.mean(nas_score_list)
    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))

    info['avg_nas_score'] = float(avg_nas_score)
    info['std_nas_score'] = float(std_nas_score)
    info['avg_precision'] = float(avg_precision)
    
    return info


'''
##################################################################################################
'''

def compute_nas_score_yolov8_v2(
    gpu, model, mixup_gamma=0.01, resolution=640, batch_size=16, repeat=32, fp16=False, verbose=True
):
    """
    改进后的 NAS 评估函数，适配 YOLOv8 模型。
    
    Args:
        gpu (int): 使用的 GPU ID。
        model (nn.Module): 需要评估的 YOLOv8 模型。
        mixup_gamma (float): Mixup 的混合比例。
        resolution (int): 输入图像的分辨率。
        batch_size (int): 每次评估的批量大小。
        repeat (int): 重复评估的次数。
        fp16 (bool): 是否使用半精度浮点数。
        verbose (bool): 是否打印详细信息。
        
    Returns:
        dict: 包含平均 NAS 分数及相关统计信息。
    """
    info = {}
    nas_score_list = []
    epsilon = 1e-8  # 小数值，用于数值稳定性
    device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')
    dtype = torch.half if fp16 else torch.float32

    model.eval()  # 设置为评估模式
    model.to(device)  # 将模型移动到设备
    if verbose:
        print(f"Model moved to {device} with dtype {dtype}")

    with torch.no_grad():
        for repeat_count in range(repeat):
            # 生成随机输入
            input1 = torch.randn([batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            input2 = torch.randn([batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            mixup_input = input1 + mixup_gamma * input2

            # 前向传播
            output1 = model(input1)
            output2 = model(mixup_input)

            # 提取特征或预测值
            features1, features2 = None, None
            if isinstance(output1, torch.Tensor):
                features1, features2 = output1, output2
            elif isinstance(output1, (list, tuple)):
                features1, features2 = output1[0], output2[0]
            elif isinstance(output1, dict):
                features1 = output1.get('pred', None) or output1.get('output', None)
                features2 = output2.get('pred', None) or output2.get('output', None)

            if features1 is None or features2 is None:
                raise ValueError("Failed to extract features. Please check the model's output format.")

            # 计算特征差异（NAS 分数的核心部分）
            if isinstance(features1, torch.Tensor):
                nas_diff = torch.abs(features1 - features2)
                nas_score = torch.sum(nas_diff, dim=tuple(range(1, features1.ndim)))
            elif isinstance(features1, (list, tuple)):
                nas_score = 0
                for feat1, feat2 in zip(features1, features2):
                    nas_diff = torch.abs(feat1 - feat2)
                    nas_score += torch.sum(nas_diff, dim=tuple(range(1, feat1.ndim)))
            else:
                raise TypeError("Unexpected feature type. Expected Tensor or list of Tensors.")

            # 对 NAS 分数取均值
            nas_score = torch.mean(nas_score)

            # 检查 NAS 分数是否有效（防止 NaN 和负数）
            if torch.isnan(nas_score) or nas_score <= 0:
                if verbose:
                    print(f"[Warning] Invalid NAS score: {nas_score.item()} at repeat {repeat_count}")
                continue

            # 计算 BatchNorm 层的缩放因子
            log_bn_scaling_factor = 0.0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var + epsilon))
                    if torch.isnan(bn_scaling_factor) or bn_scaling_factor <= 0:
                        if verbose:
                            print(f"[Warning] Invalid BN scaling factor: {bn_scaling_factor.item()}")
                        continue
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)

            # 计算最终 NAS 分数
            nas_score = torch.abs(torch.log(nas_score + epsilon)) + torch.abs(log_bn_scaling_factor)
            nas_score_list.append(float(nas_score))

    # 如果没有有效的 NAS 分数，抛出异常
    if len(nas_score_list) == 0:
        raise ValueError("No valid NAS scores were computed. Please check the model and inputs.")

    # 统计结果
    avg_nas_score = np.mean(nas_score_list)
    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))

    info['avg_nas_score'] = float(avg_nas_score)
    info['std_nas_score'] = float(std_nas_score)
    info['avg_precision'] = float(avg_precision)

    if verbose:
        print("NAS Evaluation Completed:")
        print(f"  Average NAS Score: {avg_nas_score:.4f}")
        print(f"  Standard Deviation: {std_nas_score:.4f}")
        print(f"  Average Precision (95% CI): {avg_precision:.4f}")

    return info



def compute_nas_score_yolov8_v4(
    gpu, model, mixup_gamma=0.1, resolution=640, batch_size=16, repeat=32, fp16=False, verbose=False
):
    
    info = {}
    nas_score_list = []
    multi_scale_nas_score_list = []
    epsilon = 1e-8  # 小数值，用于数值稳定性
    device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')
    dtype = torch.half if fp16 else torch.float32

    model.eval()  # 设置为评估模式
    model.to(device)  # 将模型移动到设备
    if verbose:
        print(f"Model moved to {device} with dtype {dtype}")

    # 定义钩子函数以捕获多尺度特征
    hook_outputs = {}
    target_layers = [15, 18, 21]  # P3, P4, P5 的层编号
    for i, m in enumerate(model.model):
        if i in target_layers:
            m.name = f"layer_{i}"
            m.register_forward_hook(lambda module, input, output: hook_outputs.update({module.name: output}))

    with torch.no_grad():
        for repeat_count in range(repeat):
            # 生成随机输入
            input1 = torch.randn([batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            input2 = torch.randn([batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            mixup_input = input1 + mixup_gamma * input2

            # 前向传播
            output1 = model(input1)
            output2 = model(mixup_input)

            # **计算整体输出的 NAS 分数**
            if isinstance(output1, torch.Tensor):
                features1 = output1
                features2 = output2
            elif isinstance(output1, (tuple, list)):
                features1 = output1[0]
                features2 = output2[0]
            elif isinstance(output1, dict):
                features1 = output1.get('pred', None)
                features2 = output2.get('pred', None)
                if features1 is None:
                    # 根据模型实际输出调整键名
                    features1 = output1.get('output', None)
                    features2 = output2.get('output', None)
            else:
                raise TypeError(f"Unexpected output type: {type(output1)}")

            if features1 is None or features2 is None:
                raise ValueError("Could not extract features from model outputs.")

            # 确保 features 是张量或张量的列表
            if isinstance(features1, torch.Tensor):
                # 直接计算整体 NAS 分数
                nas_diff = torch.abs(features1 - features2)
                nas_score = torch.sum(nas_diff, dim=tuple(range(1, features1.ndim)))
            elif isinstance(features1, (list, tuple)):
                # 对多个特征图进行计算
                nas_score = 0
                for feat1, feat2 in zip(features1, features2):
                    # nas_diff = torch.abs(feat1 - feat2)
                    nas_diff = torch.pow(feat1 - feat2, 2)
                    nas_score += torch.sum(nas_diff, dim=tuple(range(1, feat1.ndim)))
            else:
                raise TypeError("Features are not tensors or lists of tensors.")

            nas_score = torch.mean(nas_score)

            # 检查整体 nas_score 值
            if torch.isnan(nas_score) or nas_score <= 0:
                if verbose:
                    print(f"[Warning] Invalid overall NAS score at repeat {repeat_count}: {nas_score.item()}")
                continue

            # **计算多尺度特征的 NAS 分数**
            _ = model(input1)
            features1 = [hook_outputs[f"layer_{i}"] for i in target_layers]

            _ = model(mixup_input)
            features2 = [hook_outputs[f"layer_{i}"] for i in target_layers]

            multi_scale_nas_score = 0.0
            for i, (feat1, feat2) in enumerate(zip(features1, features2)):
                if verbose:
                    print(f"Processing scale P{i+3} with shape {feat1.shape}")

                # 计算特征图差异
                # nas_diff = torch.abs(feat1 - feat2)
                nas_diff = torch.pow(feat1 - feat2, 2)
                scale_nas_score = torch.sum(nas_diff, dim=tuple(range(1, feat1.ndim)))  # Sum over spatial dimensions
                scale_nas_score = torch.mean(scale_nas_score)  # 取均值

                # 累加多尺度 NAS 分数
                multi_scale_nas_score += scale_nas_score

            # 检查多尺度 nas_score 值
            if torch.isnan(multi_scale_nas_score) or multi_scale_nas_score <= 0:
                if verbose:
                    print(f"[Warning] Invalid multi-scale NAS score at repeat {repeat_count}: {multi_scale_nas_score.item()}")
                continue

            # 计算 BatchNorm 层的缩放因子
            log_bn_scaling_factor = 0.0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var + epsilon))
                    if torch.isnan(bn_scaling_factor) or bn_scaling_factor <= 0:
                        if verbose:
                            print(f"[Warning] Invalid BN scaling factor: {bn_scaling_factor.item()}")
                        continue
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)

            # 计算最终 NAS 分数
            final_nas_score = torch.abs(torch.log(nas_score + epsilon)) + torch.abs(log_bn_scaling_factor)
            final_multi_scale_nas_score = torch.abs(torch.log(multi_scale_nas_score + epsilon)) + torch.abs(log_bn_scaling_factor)

            # 保存结果
            nas_score_list.append(float(final_nas_score))
            multi_scale_nas_score_list.append(float(final_multi_scale_nas_score))

    # 如果没有有效的 NAS 分数，抛出异常
    if len(nas_score_list) == 0 or len(multi_scale_nas_score_list) == 0:
        raise ValueError("No valid NAS scores were computed. Please check the model and inputs.")

    # 计算统计数据
    avg_nas_score = np.mean(nas_score_list)
    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))

    avg_multi_scale_nas_score = np.mean(multi_scale_nas_score_list)
    std_multi_scale_nas_score = np.std(multi_scale_nas_score_list)
    avg_multi_scale_precision = 1.96 * std_multi_scale_nas_score / np.sqrt(len(multi_scale_nas_score_list))

    info['avg_nas_score'] = float(avg_nas_score)
    info['std_nas_score'] = float(std_nas_score)
    info['avg_precision'] = float(avg_precision)

    info['avg_multi_scale_nas_score'] = float(avg_multi_scale_nas_score)
    info['std_multi_scale_nas_score'] = float(std_multi_scale_nas_score)
    info['avg_multi_scale_precision'] = float(avg_multi_scale_precision)

    if verbose:
        print("NAS Evaluation Completed:")
        print(f"  Average Overall NAS Score: {avg_nas_score:.4f}")
        print(f"  Average Multi-Scale NAS Score: {avg_multi_scale_nas_score:.4f}")
        print(f"  Standard Deviation (Overall): {std_nas_score:.4f}")
        print(f"  Standard Deviation (Multi-Scale): {std_multi_scale_nas_score:.4f}")

    return info




if __name__ == "__main__":
    args = parse_cmd_options(sys.argv)

    # Load YOLOv8 model
    yolo_model = YOLO('/home/kongfei/code/yolov10/train_val/cfg_llm/models/v11_Poe_20_m/yolov11plus1.yaml', verbose=False) # 
    # v2: n-7.963; s-7.841; m-9.048; l-
    # v3: n-6.304;
    '''
    v4:
        n- zen-score=7.713, multi_score=6.1690;
        s- zen-score=7.867, multi_score=5.5075; 
        m- zen-score=9.258, multi_score=6.3930; 
        l- zen-score=10.46, multi_score=7.2108, zen-score=8.322, multi_score=16.6726
        x- zen-score=10.39, multi_score=7.0005; zen-score=8.334, multi_score=16.4661
        
    v5:
        n-
           
    '''
    # yolo_model = YOLO('/home/kongfei/code/yolov10/runs/detect/yolov8n/weights/best.pt') # 96.03; zen-score=96.04, multi_score=94.3048
    # yolo_model = YOLO('yolov8n.pt') # 96.03; zen-score=96.04, multi_score=94.3048
    # yolo_model = YOLO('yolov8s.pt') # 98.81; zen-score=98.81, multi_score=97.7806,
    # yolo_model = YOLO('yolov8m.pt')  # 147.7; zen-score=147.7, multi_score=146.7284
    # yolo_model = YOLO('yolov8l.pt') # 209.1; zen-score=209.1, multi_score=208.6524, 
    # yolo_model = YOLO('yolov8x.pt') # 203.5; zen-score=203.5, multi_score=204.0363,
    model = yolo_model.model  # Extract the core model
    
    if args.gpu is not None:
        model = model.cuda(args.gpu)

    # Compute NAS score
    start_timer = time.time()
    info = compute_nas_score_yolov8(
        gpu=args.gpu,
        model=model,
        mixup_gamma=0.1,
        resolution=args.input_image_size,
        batch_size=args.batch_size,
        repeat=args.repeat_times,
        fp16=False,
    )
    time_cost = (time.time() - start_timer) / args.repeat_times
    zen_score = info['avg_nas_score']
    # multi_score = info['avg_multi_scale_nas_score']
    # final_score =  info['final_score'] # final_score={final_score:.4f}
    print(info)
    # print(f'zen-score={zen_score:.4g}, multi_score={multi_score:.4f}, time cost={time_cost:.4g} second(s)')
    
    
        