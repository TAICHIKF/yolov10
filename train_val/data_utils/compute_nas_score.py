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
    # print(f"模型权重文件路径: {yolo_model.ckpt_path}")  # 打印权重路径
    # print(f"模型权重键名: {weights.keys()}")         # 打印权重键名
    print(f"模型权重: {weights}") 


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


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--input_image_size', type=int, default=640, help='Input image resolution.')
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mixup_gamma', type=float, default=1e-2)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

if __name__ == "__main__":
    args = parse_cmd_options(sys.argv)

    # Load YOLOv8 model
    # yolo_model = YOLO('yolov8.yaml') # 
    yolo_model = YOLO('yolov8n.pt') # 96.03
    # yolo_model = YOLO('yolov8s.pt') # 98.81
    # yolo_model = YOLO('yolov8m.pt')  # 147.7
    # yolo_model = YOLO('yolov8l.pt') # 209.1
    # yolo_model = YOLO('yolov8x.pt') # 203.5
    model = yolo_model.model  # Extract the core model
    

             
    if args.gpu is not None:
        model = model.cuda(args.gpu)

    # Compute NAS score
    start_timer = time.time()
    info = compute_nas_score_yolov8(
        gpu=args.gpu,
        model=model,
        mixup_gamma=args.mixup_gamma,
        resolution=args.input_image_size,
        batch_size=args.batch_size,
        repeat=args.repeat_times,
        fp16=False
    )
    time_cost = (time.time() - start_timer) / args.repeat_times
    zen_score = info['avg_nas_score']
    print(info)
    print(f'zen-score={zen_score:.4g}, time cost={time_cost:.4g} second(s)')