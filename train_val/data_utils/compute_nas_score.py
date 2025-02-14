'''
python ./train_val/data_utils/compute_nas_score.py --batch_size 16 --input_image_size 640 --repeat_times 32 --gpu 0 --mixup_gamma 0.01
'''

import sys
import torch
from torch import nn
import numpy as np
import argparse, time
from ultralytics import YOLO  # Import the YOLO model
from thop import profile  # 需要安装thop库


from ultralytics.nn.modules import Bottleneck
from ultralytics.nn.modules.block import PSABlock


# No need to import Detect if using class name checking
# from ultralytics.nn.modules.head import Detect

def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--input_image_size', type=int, default=640, help='Input image resolution.')
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=6)
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
##########################################    compute_yolov8_score  ##################################################
'''


def compute_yolov8_score(gpu, model, config, scale='n', mixup_gamma=0.01, resolutions=[640, 320], batch_size=16, repeat=16, fp16=False):
    """
    改进的NAS评分函数，结合：
    1. 模型复杂度指标（FLOPs、参数量）
    2. 多尺度特征敏感度
    3. 结构先验（来自配置文件）
    """
    device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')
    dtype = torch.half if fp16 else torch.float32
    model = model.to(device).eval()
    
    # 初始化信息字典
    info = {}
    
    # 1. 计算模型复杂度指标
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    # 2. 从配置中提取缩放因子
    depth, width, max_channels = config['scales'][scale]  # 需要传入当前模型配置
    scale_factor = depth * width  # 深度和宽度的综合指标
    
    # 3. 多分辨率特征敏感度评估
    sensitivity_scores = []
    for res in resolutions:
        # 计算特征差异
        diff_score = compute_feature_sensitivity(model, res, mixup_gamma, 
                                                batch_size, device, dtype)
        sensitivity_scores.append(diff_score)
    
    # 4. 多尺度特征融合评估（针对YOLO头部结构）
    fusion_score = evaluate_multiscale_fusion(model, device, dtype)
    
    # 5. 综合评分（需要调整权重）
    nas_score = (
        0.4 * np.log(flops/1e9) +          # FLOPs占比
        0.3 * np.log(params/1e6) +         # 参数量占比
        0.2 * np.mean(sensitivity_scores) + # 特征敏感度
        0.1 * fusion_score +               # 多尺度融合
        0.2 * scale_factor                 # 结构先验
    )
    
    info['nas_score'] = float(nas_score)
    info['flops'] = flops
    info['params'] = params
    info['sensitivity'] = np.mean(sensitivity_scores)
    return info

def compute_feature_sensitivity(model, resolution, mixup_gamma, batch_size, device, dtype):
    """计算特征敏感度得分（多尺度版本）"""
    scores = []
    with torch.no_grad():
        for _ in range(4):  # 减少重复次数
            # 生成多尺度混合输入
            base = torch.randn(batch_size, 3, resolution, resolution, device=device, dtype=dtype)
            mixed = base + mixup_gamma * torch.randn_like(base)
            
            # 获取多尺度特征
            _, features1 = model(base)   # 假设模型返回(output, features)
            _, features2 = model(mixed)
            
            # 计算各尺度特征差异
            layer_scores = []
            for f1, f2 in zip(features1, features2):
                if f1 is None or f2 is None:
                    continue
                # 使用余弦相似度度量差异
                diff = 1 - torch.cosine_similarity(f1.flatten(1), f2.flatten(1)).mean()
                layer_scores.append(float(diff))
            scores.append(np.mean(layer_scores))
    return np.mean(scores)


# 通用特征敏感度计算
def compute_unified_sensitivity(model, resolution, mixup_gamma, batch_size, 
                               device, dtype, feature_extractor, diff_method):
    scores = []
    with torch.no_grad():
        for _ in range(4):
            base = torch.randn(batch_size, 3, resolution, resolution, 
                              device=device, dtype=dtype)
            mixed = base + mixup_gamma * torch.randn_like(base)
            
            features1 = feature_extractor(model(base))
            features2 = feature_extractor(model(mixed))
            
            diffs = []
            for f1, f2 in zip(features1, features2):
                if f1 is None or f2 is None:
                    continue
                
                if diff_method == 'cosine':
                    diff = 1 - torch.cosine_similarity(f1.flatten(1), f2.flatten(1)).mean()
                elif diff_method == 'l2':
                    diff = torch.norm(f1 - f2, p=2) / (torch.norm(f1, p=2) + 1e-8)
                
                diffs.append(float(diff))
            
            if diffs:
                scores.append(np.mean(diffs))
    
    return np.mean(scores) if scores else 0


def evaluate_multiscale_fusion(model, device, dtype):
    """评估多尺度特征融合效果"""
    with torch.no_grad():
        input_tensor = torch.randn(1, 3, 640, 640).to(device)
        _, features = model(input_tensor)  # 获取多尺度特征
        
        # 计算特征图之间的相关性
        fusion_score = 0
        for i in range(len(features)-1):
            # 上采样并计算特征相似度
            up_feat = nn.functional.interpolate(features[i+1], scale_factor=2)
            sim = torch.cosine_similarity(features[i].flatten(), up_feat.flatten(), dim=0)
            fusion_score += float(sim)
    return fusion_score / (len(features)-1)



############################################# yolov11的评分计算  #############################################
# ---------- 将辅助函数移到外部 ----------
def torch_histc2d(x, y, bins=32, min=(-3, -3), max=(3, 3)):
    """PyTorch实现的二维直方图"""
    device = x.device
    x = x.clamp(min[0], max[0])
    y = y.clamp(min[1], max[1])
    
    # 计算bin索引
    x_bin = ((x - min[0]) / (max[0] - min[0]) * bins).floor().long().clamp(0, bins-1)
    y_bin = ((y - min[1]) / (max[1] - min[1]) * bins).floor().long().clamp(0, bins-1)
    
    # 使用一维索引
    linear_indices = x_bin * bins + y_bin
    hist = torch.bincount(linear_indices, minlength=bins*bins).view(bins, bins)
    return hist.float()

def mutual_info_continuous(x, y, bins=32):
    """修复后的互信息计算"""
    # 标准化
    x = (x - x.mean()) / (x.std() + 1e-8)
    y = (y - y.mean()) / (y.std() + 1e-8)
    
    # 计算直方图
    hist_x = torch.histc(x, bins=bins, min=-3, max=3) + 1e-8
    hist_y = torch.histc(y, bins=bins, min=-3, max=3) + 1e-8
    hist_2d = torch_histc2d(x, y, bins=bins) + 1e-8
    
    # 计算概率
    p_x = hist_x / hist_x.sum()
    p_y = hist_y / hist_y.sum()
    p_xy = hist_2d / hist_2d.sum()
    
    # 计算互信息
    mi = (p_xy * (torch.log(p_xy) - torch.log(p_x.unsqueeze(1)) - torch.log(p_y.unsqueeze(0)))).sum()
    return mi.item()

def compute_psa_score(feat):
    """通道相关性计算（独立函数）"""
    if feat.dim() != 4 or feat.size(1) < 2:
        return 0.0
    
    C = feat.size(1)
    feat_flat = feat.permute(1,0,2,3).flatten(1)
    
    # 协方差计算
    cov = torch.cov(feat_flat)
    diag = torch.diag(cov) + 1e-8
    norm_cov = cov / torch.sqrt(torch.outer(diag, diag))
    
    # 排除对角线
    mask = ~torch.eye(C, dtype=torch.bool, device=feat.device)
    return torch.mean(torch.abs(norm_cov[mask])).item()

def yolov11_fusion_eval(model, device):
    """独立融合评估函数"""
    with torch.no_grad():
        input_tensor = torch.randn(1, 3, 640, 640).to(device)
        _, features = model(input_tensor)
        
        # PSA评分
        psa_scores = [compute_psa_score(f) for f in features if f is not None]
        valid_psa = [s for s in psa_scores if not np.isnan(s)]
        final_psa = np.mean(valid_psa) if valid_psa else 0
        
        # 融合评分
        fusion_scores = []
        for i in range(len(features)-1):
            if features[i] is None or features[i+1] is None:
                continue
            
            try:
                up_feat = nn.functional.interpolate(
                    features[i+1], 
                    size=features[i].shape[2:], 
                    mode='nearest'
                )
                mi = mutual_info_continuous(features[i].flatten(), up_feat.flatten())
                if not np.isnan(mi):
                    fusion_scores.append(mi)
            except Exception as e:
                print(f"融合计算跳过: {str(e)}")
        
        final_fusion = np.mean(fusion_scores) if fusion_scores else 0
        
        return 0.7 * final_fusion + 0.3 * final_psa


# ---------- 主函数修改 ----------
def compute_yolov11_score(gpu, model, config, scale, mixup_gamma=0.01, resolutions=[640], batch_size=16, repeat=16, fp16=False):
    device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')
    model = model.to(device).eval()
    
    # 1. 模型复杂度
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    # 2. 结构先验
    depth, width, max_channels = config['scales'][scale]
    effective_depth = depth * 0.6  # 根据实际层数调整
    scale_factor = (effective_depth ** 0.5) * (width ** 0.8)
    
    # 3. 特征敏感度
    def feature_extractor(output):
        return [f.detach() for f in output[1]] if output[1] else []
    
    sensitivity_scores = [
        compute_feature_sensitivity_v11(
            model, res, mixup_gamma, batch_size, 
            device, torch.half if fp16 else torch.float32,
            feature_extractor
        )
        for res in resolutions
    ]
    
    # 4. 融合评估
    fusion_score = yolov11_fusion_eval(model, device)
    
    # 5. 综合评分
    nas_score = (
        0.35 * np.log(flops/1e9) +
        0.25 * np.log(params/1e6) +
        0.25 * np.mean(sensitivity_scores) +
        0.15 * fusion_score +
        0.3 * scale_factor
    )
    
    return {
        'nas_score': float(nas_score),
        'flops': flops,
        'params': params
    }

def compute_feature_sensitivity_v11(model, resolution, mixup_gamma, batch_size, device, dtype, feature_extractor):
    """特征敏感度计算"""
    scores = []
    with torch.no_grad():
        for _ in range(4):
            base = torch.randn(batch_size, 3, resolution, resolution, device=device, dtype=dtype)
            mixed = base + mixup_gamma * torch.randn_like(base)
            
            features1 = feature_extractor(model(base))
            features2 = feature_extractor(model(mixed))
            
            diffs = []
            for f1, f2 in zip(features1, features2):
                if f1 is None or f2 is None:
                    continue
                diff = torch.norm(f1 - f2, p=2) / (torch.norm(f1, p=2) + 1e-8)
                diffs.append(diff.item())
            
            if diffs:
                scores.append(np.mean(diffs))
    
    return np.mean(scores) if scores else 0



##################################################################################################
# 混合 通用版本
def compute_attention_effect(model, device):
    """评估注意力模块的有效性"""
    attn_scores = []
    for name, module in model.named_modules():
        if isinstance(module, PSABlock):
            # 计算注意力权重熵
            with torch.no_grad():
                x = torch.randn(1, module.dim, 32, 32).to(device)
                attn_weights = module.get_attention_map(x)  # 假设能获取注意力图
                entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
                attn_scores.append(entropy.mean().item())
    return np.mean(attn_scores) if attn_scores else 0


def evaluate_kernel_effect(model):
    """评估不同卷积核尺寸的影响"""
    kernel_scores = []
    for name, module in model.named_modules():
        if isinstance(module, Bottleneck) and hasattr(module.conv, 'kernel_size'):
            k = module.conv.kernel_size[0]
            # 大卷积核赋予更高权重（假设能捕获更多上下文）
            kernel_scores.append(k / 3)  # 3x3得1, 5x5得1.67
    return np.mean(kernel_scores) if kernel_scores else 1


def evaluate_sppf_diversity(model, device):
    """评估SPPF特征多样性"""
    with torch.no_grad():
        x = torch.randn(1, 3, 640, 640).to(device)
        features = model.backbone(x)  # 获取SPPF层输出
        sppf_feat = features[-1]
        
        # 计算不同池化路径的差异
        pooled = [F.avg_pool2d(sppf_feat, k) for k in [5, 9, 13]]
        diffs = [torch.norm(pooled[i]-pooled[j]).item() 
                for i in range(3) for j in range(i+1,3)]
    return np.mean(diffs)


# 两种融合评估方法
def cosine_fusion_eval(model, device):
    """通用余弦相似度融合评估"""
    with torch.no_grad():
        input_tensor = torch.randn(1, 3, 640, 640).to(device)
        output = model(input_tensor)
        features = output[1] if isinstance(output, tuple) else [output]
        
        fusion_score = 0
        valid_pairs = 0
        for i in range(len(features)-1):
            up_feat = nn.functional.interpolate(features[i+1], scale_factor=2, mode='nearest')
            sim = torch.cosine_similarity(features[i].flatten(), up_feat.flatten(), dim=0)
            fusion_score += float(sim)
            valid_pairs += 1
        
        return fusion_score / valid_pairs if valid_pairs > 0 else 0
    
def mi_psa_fusion_eval(model, device):
    """YOLOv11专用融合评估"""
    with torch.no_grad():
        input_tensor = torch.randn(1, 3, 640, 640).to(device)
        _, features = model(input_tensor)
        
        # PSA评分
        psa_scores = [compute_psa_score(f) for f in features if f is not None]
        valid_psa = [s for s in psa_scores if not np.isnan(s)]
        final_psa = np.mean(valid_psa) if valid_psa else 0
        
        # 互信息评分
        fusion_scores = []
        for i in range(len(features)-1):
            if features[i] is None or features[i+1] is None:
                continue
            
            try:
                up_feat = nn.functional.interpolate(
                    features[i+1], 
                    size=features[i].shape[2:], 
                    mode='nearest'
                )
                mi = mutual_info_continuous(features[i].flatten(), up_feat.flatten())
                if not np.isnan(mi):
                    fusion_scores.append(mi)
            except Exception as e:
                print(f"融合计算跳过: {str(e)}")
        
        final_fusion = np.mean(fusion_scores) if fusion_scores else 0
        
        return 0.7 * final_fusion + 0.3 * final_psa    
    

def compute_yolo_score(gpu, model, config, model_type='v8', scale='n', 
                      mixup_gamma=0.01, resolutions=[640, 320], batch_size=16, 
                      repeat=16, fp16=False):
    """
    通用YOLO评分函数，支持v8/v11等版本
    model_type: 'v8' 或 'v11'
    """
    device = torch.device(f'cuda:{gpu}' if gpu is not None else 'cpu')
    dtype = torch.half if fp16 else torch.float32
    model = model.to(device).eval()
    
    # 初始化评分组件配置
    config_ = {
        'v8': {
            'structure_formula': lambda d,w: d * w,
            'diff_method': 'cosine',
            'fusion_eval': 'cosine',
            'weights': [0.4, 0.3, 0.2, 0.1, 0.2]
        },
        'v11': {
            # 'structure_formula': lambda d,w: (d*0.6)**0.5 * w**0.8,
            # 'diff_method': 'l2',
            # 'fusion_eval': 'mi_psa',
            # 'weights': [0.35, 0.25, 0.25, 0.15, 0.3]
            'structure_formula': lambda d,w: d * w,
            'diff_method': 'cosine',
            'fusion_eval': 'cosine',
            'weights': [0.4, 0.3, 0.2, 0.1, 0.2]
            
        }
    }[model_type]
    
    # 1. 模型复杂度指标
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    # 2. 结构先验计算
    depth, width, _ = config['scales'][scale]
    scale_factor = config_['structure_formula'](depth, width)
    
    # 3. 特征敏感度评估
    def feature_extractor(output):
        return output[1] if isinstance(output, tuple) else [output]
    
    sensitivity_scores = []
    for res in resolutions:
        score = compute_unified_sensitivity(
            model, res, mixup_gamma, batch_size, device, dtype,
            feature_extractor=feature_extractor,
            diff_method=config_['diff_method']
        )
        sensitivity_scores.append(score)
    
    # 4. 多尺度融合评估
    if config_['fusion_eval'] == 'cosine':
        fusion_score = cosine_fusion_eval(model, device)
    else:
        fusion_score = mi_psa_fusion_eval(model, device)
    
    # 5. 综合评分
    w = config_['weights']
    nas_score = (
        w[0] * np.log(flops/1e9) +
        w[1] * np.log(params/1e6) +
        w[2] * np.mean(sensitivity_scores) +
        w[3] * fusion_score +
        w[4] * scale_factor
    )
    
    return {
        'nas_score': float(nas_score),
        'flops': flops,
        'params': params,
        'sensitivity': np.mean(sensitivity_scores),
        'fusion': fusion_score,
        'structure': scale_factor
    }





if __name__ == "__main__":
    args = parse_cmd_options(sys.argv)


    # Compute NAS score
    
    # info = compute_nas_score_yolov8(
    #     gpu=args.gpu,
    #     model=model,
    #     mixup_gamma=0.1,
    #     resolution=args.input_image_size,
    #     batch_size=args.batch_size,
    #     repeat=args.repeat_times,
    #     fp16=False,
    # )
    
    
    #  使用示例
    
    
    
    
    
    yolov8_config = {
        'scales': {
            'n': [0.33, 0.25, 1024], 
            's': [0.33, 0.50, 1024],  
            'm': [0.67, 0.75, 768],  
            'l': [1.00, 1.00, 512],  
            'x': [1.00, 1.25, 512],  
        }
    }
    
    
    yolo_model = YOLO(f'/home/kongfei/code/yolov10/train_val/cfg_llm/model/Poe_10__/yolov8plus10.yaml', verbose=False)  # 
    scale = 'n'
    version = 'v8'
    model = yolo_model.model  # Extract the core model
    score_info1 = compute_yolov8_score(args.gpu, model, yolov8_config, scale)
    score_info2= compute_yolo_score(args.gpu, model, yolov8_config, version, scale)
    score_info3 = compute_yolov11_score(args.gpu, model, yolov8_config, scale)
    print(f"{scale}", score_info1['nas_score'], "time_cost:", )
    print(f"{scale}", score_info2['nas_score'], "time_cost:", )
    print(f"{scale}", score_info3['nas_score'], "time_cost:", )
    
    # for scale in yolov8_config['scales']:
        
    #     version = 'v8'
    
    #     start_timer = time.time()
    #     # scale = 'n'
    #     # Load YOLO model
    #     # model_name = f'/home/kongfei/code/yolov10/yolov8{scale}.pt'
    #     # yolo_model = YOLO(model_name, verbose=True)  # 
    #     yolo_model = YOLO(f'yolov8{scale}.yaml', verbose=False)  # 
        
    #     model = yolo_model.model  # Extract the core model
        
    #     if args.gpu is not None:
    #         model = model.cuda(args.gpu)
        
    #     score_info1 = compute_yolov8_score(args.gpu, model, yolov8_config, scale)
    #     score_info2= compute_yolo_score(args.gpu, model, yolov8_config, version, scale)
    #     score_info3 = compute_yolov11_score(args.gpu, model, yolov8_config, scale)
        
    #     time_cost = (time.time() - start_timer) / args.repeat_times
    #     # zen_score = score_info1['nas_score']

    #     print(f"{scale}", score_info1['nas_score'], "time_cost:", time_cost)
    #     print(f"{scale}", score_info2['nas_score'], "time_cost:", time_cost)
    #     print(f"{scale}", score_info3['nas_score'], "time_cost:", time_cost)
    
    
    # yolov11_config = {
    #     'scales': {
    #         'n': [0.50, 0.25, 1024],  # summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs, 39.5 mAP(val50-95)
    #         's': [0.50, 0.50, 1024],  # summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs, 47.0 mAP(val50-95)
    #         'm': [0.50, 1.00, 512],   # summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs, 51.5 mAP(val50-95)
    #         'l': [1.00, 1.00, 512],   # summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs, 53.4 mAP(val50-95)
    #         'x': [1.00, 1.50, 512],   # summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs, 54.7 mAP(val50-95)
    #     }
    # }
    
    # print("yolov11:")
    # for scale in yolov11_config['scales']:
    
    #     version = 'v11'
    #     start_timer = time.time()
    #     # scale = 'n'
    #     # Load YOLO model
    #     # model_name = f'/home/kongfei/code/yolov10/yolo11{scale}.pt'
    #     # yolo_model = YOLO(model_name, verbose=True)  # 

    #     # yolo_model = YOLO('/home/kongfei/code/yolov10/train_val/cfg_llm/models/yolov11.yaml', verbose=True)  # 
    #     yolo_model = YOLO(f'yolov11{scale}.yaml', verbose=False)  # 
        
    #     model = yolo_model.model  # Extract the core model
        
    #     if args.gpu is not None:
    #         model = model.cuda(args.gpu)
        
    #     score_info1 = compute_yolov8_score(args.gpu, model, yolov8_config, scale)
    #     score_info2= compute_yolo_score(args.gpu, model, yolov8_config, version, scale)
    #     score_info3 = compute_yolov11_score(args.gpu, model, yolov8_config, scale)
        
    #     time_cost = (time.time() - start_timer) / args.repeat_times
    #     # zen_score = score_info1['nas_score']

    #     time_cost = (time.time() - start_timer) / args.repeat_times
    #     print(f"{scale}", score_info1['nas_score'], "time_cost:", time_cost)
    #     print(f"{scale}", score_info2['nas_score'], "time_cost:", time_cost)
    #     print(f"{scale}", score_info3['nas_score'], "time_cost:", time_cost)
    
  

 
    