from openai import OpenAI
from zhipuai import ZhipuAI
import re
import time
import logging


system_content = "You are Quoc V. Le, a computer scientist and artificial intelligence researcher who is widely regarded as one of the leading experts in deep learning and neural network architecture search.  Your work in this area has focused on developing efficient algorithms for searching the space of possible neural network architectures, with the goal of finding architectures that perform well on a given task while minimizing the computational cost of training and inference."

# YAML 内容作为多行字符串
yolov8_config_yaml = """
# Optimized YOLOv8 object detection model with P3-P5 outputs. Targeting performance improvement without increasing parameters.

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
  s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
  m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
  l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
  x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs

# YOLOv8n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9

# YOLOv8n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2f, [1024]] # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)
"""

yolov11_config_yaml = """
# YOLO11 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
  m: [0.50, 1.00, 512] # summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [1.00, 1.00, 512] # summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [1.00, 1.50, 512] # summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLO11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)

  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)
"""

modules = "['Classify','Conv','ConvTranspose','GhostConv','Bottleneck','GhostBottleneck','SPP','SPPF','C2fPSA','C2PSA','DWConv','Focus','BottleneckCSP','C1,'C2','C2f','C3k2','RepNCSPELAN4','ELAN1','ADown','AConv','SPPELAN','C2fAttn','C3,'C3TR','C3Ghost','nn.ConvTranspose2d','DWConvTranspose2d','C3x','RepC3','PSA','SCDown','C2fCIB']"


suffix = '''Please do not include anything else other than configuration in your response!'''

# user_input = f'''You need to analyze where yolov11 is better than yolov8, and then understand and improve on the basis of yolov8 to make the newly generated configuration better than yolov8. The configuration file for yolov8 is {yolov8_config_yaml}, The configuration file for yolov11 is{yolov11_config_yaml}'''
user_input = f'''You need to analyze yolov8 to make the newly generated configuration better than yolov8. The configuration file for yolov8 is {yolov8_config_yaml}'''


def generate_new_structure_using_llm(api_type):
    
    if api_type == 'gpt' :
        api_key = 'sk-N4zU8nzRn6ifYsbgs0N4UGCHDqX5a6g6AI0OBsSfhk2AuHoV'
    elif api_type == 'glm':
        api_key = 'd374a99b15210255c4ec14118b86c3b7.uESar1mpd1z0Eu9w'
    elif api_type == 'qwen':
        api_key = 'sk-9b820f1968304452a41b477ae75ff735'
    elif api_type == 'kimi':
        api_key = 'sk-6goEINStkkpauoOjXyi8FmphOoyrI8U06ATAFBacMo8o8ttA'
    elif api_type == 'silicon':
        api_key = 'sk-iwupklctuhgbqpxpyxwnlvykkdvvnosvzgiwygukdntxkibp'
    else:
        api_key = None
        print("# Api key error!")
    
    # prompt_cn = '根据现有配置，生成一个新的优化后的配置，优化目标：参数量不增加或参数量减少情况下提升目标检测性能。具体的module顺序你可以更改，channel数值也可以变化。总之，你可以根据自己的理解生成新结构，结果比原有配置性能更优就可以。'
    prompt = '''Generate a new optimized configuration based on the existing configuration.
             Optimization goal: Improve target detection performance when the number of parameters does not increase or decreases. 
             You can change the specific module order, and the channel value can also change. However, modules can only be types in {} and cannot be generated randomly. Sizes of tensors must match except in dimension 1.
             In short, you can generate a new structure according to your own understanding, and the result is better than the original configuration.'''
             
             
    messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_input + prompt + suffix},
        ] 


    if api_type == 'gpt' :
        # 配置 OpenAI API
        client = OpenAI(api_key=api_key, base_url="https://api.chatanywhere.tech/v1")
        response = client.chat.completions.create(
            # model='gpt-4',  # 使用适当的模型名称
            model='gpt-4o-mini',  # 
            # model='gpt-3.5-turbo',  #     (200 / per day)
            messages=messages,
            temperature=1,
            # n=1
        )


    elif api_type == 'glm':
        client = ZhipuAI(api_key=api_key) # 请填写您自己的APIKey
        response = client.chat.completions.create(
        # model="glm-4-0520",  
        model="glm-4-long",
        # model="glm-4-Flash", 
        messages=messages,
        temperature=1, 
        )


    elif api_type == 'qwen':
        client = OpenAI(
                api_key=api_key, # 如果您没有配置环境变量，请在此处用您的API Key进行替换
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
            )
        # time.sleep(5)
        response = client.chat.completions.create(
                # model="qwen1.5-7b-chat",  # rank 6
                # model="qwen1.5-110b-chat",  # 226
                # model="qwen2-0.5b-instruct",  # fail
                # model="qwen2-1.5b-instruct",  # rank 1029
                model="qwen2-72b-instruct", #  110（score）
                # model="qwen2.5-7b-instruct",  # 
                # model="qwen2.5-14b-instruct",  # 4
                # model="qwen2.5-32b-instruct",  # 80.43 / 20;  10 /min
                # model="qwen2.5-72b-instruct",  #  99.60  60 / min    
                # model="qwen2.5-coder-1.5b-instruct",  #  99.60  60 / min    
                # model="qwen2.5-coder-7b-instruct",  #  99.60  60 / min    
                messages=messages,
                temperature=1, 
            ) 
        
    elif api_type == "silicon":        
        client = OpenAI(
            api_key=api_key,
            base_url='https://api.siliconflow.cn/v1',
        )
        response = client.chat.completions.create(
            # model="deepseek-ai/DeepSeek-V2.5",
            model="Qwen/Qwen2.5-7B-Instruct",
            # model="THUDM/glm-4-9b-chat",
            # model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            messages=messages,
            # stream=False  # 启用流式输出
        )

    elif api_type =='kimi':
        time.sleep(20)
        client = OpenAI(
                api_key=api_key, # 在这里将 MOONSHOT_API_KEY 替换为你从 Kimi 开放平台申请的 API Key
                base_url="https://api.moonshot.cn/v1",
            )
        response = client.chat.completions.create(
                # model = "moonshot-v1-8k",
                model = "moonshot-v1-128k",
                messages = messages,
                temperature = 0.9,
            )


    # logging.info(f'# response:{response}')
    # print(f'# response:{response}')
    # 获取响应内容
    new_structure = response.choices[0].message.content
    
    
    # 去除 ```yaml 和 ```
    cleaned_new_yaml = new_structure.strip("```yaml").strip("```")

    
    return cleaned_new_yaml