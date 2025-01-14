from openai import OpenAI
from zhipuai import ZhipuAI
import time
import requests
from .yolo_yaml import clean_markdown_yaml


yolov11n_config_yaml = """
# Optimized YOLOv11 object detection model with P3-P5 outputs. Targeting performance improvement without increasing parameters.

# Parameters
nc: 80  # number of classes
scales:  # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]  # YOLOv11n summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024]  # YOLOv11s summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
  m: [0.50, 1.00, 512]  # YOLOv11m summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [1.00, 1.00, 512]  # YOLOv11l summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [1.00, 1.50, 512]  # YOLOv11x summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLOv11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2， 第0层，Conv模块，输入为原始图像(640x640x3)，卷积参数：输出通道数为64，卷积核大小为3x3，步长为2。
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4， 第1层，Conv模块，输入为上一层输出，输出通道数为128，卷积核大小为3x3，步长为2。
  - [-1, 2, C2f, [128, True]]  # 2， 第2层，C2f模块，重复2次，输出通道数为128，带有shortcut连接。
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8， 第3层，Conv模块，输入为上一层输出，输出通道数为256，卷积核大小为3x3，步长为2。
  - [-1, 4, C2f, [256, True]]  # 4，  第4层，C2f模块，重复4次，输出通道数为256，带有shortcut连接。
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16， 第5层，Conv模块，输入为上一层输出，输出通道数为512，卷积核大小为3x3，步长为2。
  - [-1, 6, C2f, [512, True]]  # 6 ， 第6层，C2f模块，重复6次，输出通道数为512，带有shortcut连接。
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32， 第7层，Conv模块，输入为上一层输出，输出通道数为1024，卷积核大小为3x3，步长为2。
  - [-1, 3, C2f, [1024, True]]  # 8， 第8层，C2f模块，重复3次，输出通道数为1024，带有shortcut连接。
  - [-1, 1, SPPF, [1024, 5]]  # 9， 第9层，SPPF模块（快速空间金字塔池化层），输出通道数为1024，池化核大小为5。

# YOLOv11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 10， 第10层，上采样层，输入为上一层输出，scale_factor=2，mode=nearest。
  - [[-1, 6], 1, Concat, [1]]  # 11-cat backbone P4， 第11层，Concat模块，将第10层和第6层的输出按通道拼接。
  - [-1, 3, C2f, [512]]  # 12， 第12层，C2f模块，重复3次，输出通道数为512，shortcut关闭。
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 13，第13层，上采样层，scale_factor=2，mode=nearest。
  - [[-1, 4], 1, Concat, [1]]  # 14-cat backbone P3， 第14层，Concat模块，将第13层和第4层的输出按通道拼接。
  - [-1, 3, C2f, [256]]  # 15-P3/8 (small) 第15层，C2f模块，重复3次，输出通道数为256，shortcut关闭。
  - [-1, 1, Conv, [256, 3, 2]]  # 16  第16层，Conv模块，步长为2。
  - [[-1, 12], 1, Concat, [1]]  # 17-cat head P4，第17层，Concat模块，将第16层和第12层的输出按通道拼接。
  - [-1, 3, C2f, [512]]  # 18-P4/16 (medium) ，第18层，C2f模块，重复3次，输出通道数为512，shortcut关闭。
  - [-1, 1, Conv, [512, 3, 2]]  # 19  第19层，Conv模块，步长为2。
  - [[-1, 9], 1, Concat, [1]]  # 20-cat head P5 ，第20层，Concat模块，将第19层和第9层的输出按通道拼接。
  - [-1, 3, C2f, [1024]]  # 21-P5/32 (large) ，第21层，C2f模块，重复3次，输出通道数为1024，shortcut关闭。
  - [[15, 18, 21], 1, Detect, [nc]]  # 22-Detect(P3, P4, P5) ，第22层，Detect模块，输入为第15层（P3）、第18层（P4）和第21层（P5）的输出。
"""
# YAML 内容作为多行字符串
yolov11s_config_yaml = """
# Optimized YOLOv11 object detection model with P3-P5 outputs. Targeting performance improvement without increasing parameters.

# Parameters
nc: 80  # number of classes
scales:  # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]  # YOLOv11n summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024]  # YOLOv11s summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
  m: [0.50, 1.00, 512]  # YOLOv11m summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [1.00, 1.00, 512]  # YOLOv11l summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [1.00, 1.50, 512]  # YOLOv11x summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLOv11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2， 第0层，Conv模块，输入为原始图像(640x640x3)，卷积参数：输出通道数为64，卷积核大小为3x3，步长为2。
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4， 第1层，Conv模块，输入为上一层输出，输出通道数为128，卷积核大小为3x3，步长为2。
  - [-1, 2, C2f, [128, True]]  # 2， 第2层，C2f模块，重复2次，输出通道数为128，带有shortcut连接。
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8， 第3层，Conv模块，输入为上一层输出，输出通道数为256，卷积核大小为3x3，步长为2。
  - [-1, 4, C2f, [256, True]]  # 4，  第4层，C2f模块，重复4次，输出通道数为256，带有shortcut连接。
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16， 第5层，Conv模块，输入为上一层输出，输出通道数为512，卷积核大小为3x3，步长为2。
  - [-1, 6, C2f, [512, True]]  # 6 ， 第6层，C2f模块，重复6次，输出通道数为512，带有shortcut连接。
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32， 第7层，Conv模块，输入为上一层输出，输出通道数为1024，卷积核大小为3x3，步长为2。
  - [-1, 3, C2f, [1024, True]]  # 8， 第8层，C2f模块，重复3次，输出通道数为1024，带有shortcut连接。
  - [-1, 1, SPPF, [1024, 5]]  # 9， 第9层，SPPF模块（快速空间金字塔池化层），输出通道数为1024，池化核大小为5。

# YOLOv11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 10， 第10层，上采样层，输入为上一层输出，scale_factor=2，mode=nearest。
  - [[-1, 6], 1, Concat, [1]]  # 11-cat backbone P4， 第11层，Concat模块，将第10层和第6层的输出按通道拼接。
  - [-1, 3, C2f, [512]]  # 12， 第12层，C2f模块，重复3次，输出通道数为512，shortcut关闭。
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 13，第13层，上采样层，scale_factor=2，mode=nearest。
  - [[-1, 4], 1, Concat, [1]]  # 14-cat backbone P3， 第14层，Concat模块，将第13层和第4层的输出按通道拼接。
  - [-1, 3, C2f, [256]]  # 15-P3/8 (small) 第15层，C2f模块，重复3次，输出通道数为256，shortcut关闭。
  - [-1, 1, Conv, [256, 3, 2]]  # 16  第16层，Conv模块，步长为2。
  - [[-1, 12], 1, Concat, [1]]  # 17-cat head P4，第17层，Concat模块，将第16层和第12层的输出按通道拼接。
  - [-1, 3, C2f, [512]]  # 18-P4/16 (medium) ，第18层，C2f模块，重复3次，输出通道数为512，shortcut关闭。
  - [-1, 1, Conv, [512, 3, 2]]  # 19  第19层，Conv模块，步长为2。
  - [[-1, 9], 1, Concat, [1]]  # 20-cat head P5 ，第20层，Concat模块，将第19层和第9层的输出按通道拼接。
  - [-1, 3, C2f, [1024]]  # 21-P5/32 (large) ，第21层，C2f模块，重复3次，输出通道数为1024，shortcut关闭。
  - [[15, 18, 21], 1, Detect, [nc]]  # 22-Detect(P3, P4, P5) ，第22层，Detect模块，输入为第15层（P3）、第18层（P4）和第21层（P5）的输出。
"""
# YAML 内容作为多行字符串
yolov11m_config_yaml = """
# Optimized YOLOv11 object detection model with P3-P5 outputs. Targeting performance improvement without increasing parameters.

# Parameters
nc: 80  # number of classes
scales:  # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]  # YOLOv11n summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024]  # YOLOv11s summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
  m: [0.50, 1.00, 512]  # YOLOv11m summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [1.00, 1.00, 512]  # YOLOv11l summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [1.00, 1.50, 512]  # YOLOv11x summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLOv11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2， 第0层，Conv模块，输入为原始图像(640x640x3)，卷积参数：输出通道数为64，卷积核大小为3x3，步长为2。
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4， 第1层，Conv模块，输入为上一层输出，输出通道数为128，卷积核大小为3x3，步长为2。
  - [-1, 2, C2f, [128, True]]  # 2， 第2层，C2f模块，重复2次，输出通道数为128，带有shortcut连接。
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8， 第3层，Conv模块，输入为上一层输出，输出通道数为256，卷积核大小为3x3，步长为2。
  - [-1, 4, C2f, [256, True]]  # 4，  第4层，C2f模块，重复4次，输出通道数为256，带有shortcut连接。
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16， 第5层，Conv模块，输入为上一层输出，输出通道数为512，卷积核大小为3x3，步长为2。
  - [-1, 6, C2f, [512, True]]  # 6 ， 第6层，C2f模块，重复6次，输出通道数为512，带有shortcut连接。
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32， 第7层，Conv模块，输入为上一层输出，输出通道数为1024，卷积核大小为3x3，步长为2。
  - [-1, 3, C2f, [1024, True]]  # 8， 第8层，C2f模块，重复3次，输出通道数为1024，带有shortcut连接。
  - [-1, 1, SPPF, [1024, 5]]  # 9， 第9层，SPPF模块（快速空间金字塔池化层），输出通道数为1024，池化核大小为5。

# YOLOv11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 10， 第10层，上采样层，输入为上一层输出，scale_factor=2，mode=nearest。
  - [[-1, 6], 1, Concat, [1]]  # 11-cat backbone P4， 第11层，Concat模块，将第10层和第6层的输出按通道拼接。
  - [-1, 3, C2f, [512]]  # 12， 第12层，C2f模块，重复3次，输出通道数为512，shortcut关闭。
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 13，第13层，上采样层，scale_factor=2，mode=nearest。
  - [[-1, 4], 1, Concat, [1]]  # 14-cat backbone P3， 第14层，Concat模块，将第13层和第4层的输出按通道拼接。
  - [-1, 3, C2f, [256]]  # 15-P3/8 (small) 第15层，C2f模块，重复3次，输出通道数为256，shortcut关闭。
  - [-1, 1, Conv, [256, 3, 2]]  # 16  第16层，Conv模块，步长为2。
  - [[-1, 12], 1, Concat, [1]]  # 17-cat head P4，第17层，Concat模块，将第16层和第12层的输出按通道拼接。
  - [-1, 3, C2f, [512]]  # 18-P4/16 (medium) ，第18层，C2f模块，重复3次，输出通道数为512，shortcut关闭。
  - [-1, 1, Conv, [512, 3, 2]]  # 19  第19层，Conv模块，步长为2。
  - [[-1, 9], 1, Concat, [1]]  # 20-cat head P5 ，第20层，Concat模块，将第19层和第9层的输出按通道拼接。
  - [-1, 3, C2f, [1024]]  # 21-P5/32 (large) ，第21层，C2f模块，重复3次，输出通道数为1024，shortcut关闭。
  - [[15, 18, 21], 1, Detect, [nc]]  # 22-Detect(P3, P4, P5) ，第22层，Detect模块，输入为第15层（P3）、第18层（P4）和第21层（P5）的输出。
"""
# YAML 内容作为多行字符串
yolov11l_config_yaml = """
# Optimized YOLOv11 object detection model with P3-P5 outputs. Targeting performance improvement without increasing parameters.

# Parameters
nc: 80  # number of classes
scales:  # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]  # YOLOv11n summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024]  # YOLOv11s summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
  m: [0.50, 1.00, 512]  # YOLOv11m summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [1.00, 1.00, 512]  # YOLOv11l summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [1.00, 1.50, 512]  # YOLOv11x summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLOv11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2， 第0层，Conv模块，输入为原始图像(640x640x3)，卷积参数：输出通道数为64，卷积核大小为3x3，步长为2。
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4， 第1层，Conv模块，输入为上一层输出，输出通道数为128，卷积核大小为3x3，步长为2。
  - [-1, 2, C2f, [128, True]]  # 2， 第2层，C2f模块，重复2次，输出通道数为128，带有shortcut连接。
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8， 第3层，Conv模块，输入为上一层输出，输出通道数为256，卷积核大小为3x3，步长为2。
  - [-1, 4, C2f, [256, True]]  # 4，  第4层，C2f模块，重复4次，输出通道数为256，带有shortcut连接。
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16， 第5层，Conv模块，输入为上一层输出，输出通道数为512，卷积核大小为3x3，步长为2。
  - [-1, 6, C2f, [512, True]]  # 6 ， 第6层，C2f模块，重复6次，输出通道数为512，带有shortcut连接。
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32， 第7层，Conv模块，输入为上一层输出，输出通道数为1024，卷积核大小为3x3，步长为2。
  - [-1, 3, C2f, [1024, True]]  # 8， 第8层，C2f模块，重复3次，输出通道数为1024，带有shortcut连接。
  - [-1, 1, SPPF, [1024, 5]]  # 9， 第9层，SPPF模块（快速空间金字塔池化层），输出通道数为1024，池化核大小为5。

# YOLOv11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 10， 第10层，上采样层，输入为上一层输出，scale_factor=2，mode=nearest。
  - [[-1, 6], 1, Concat, [1]]  # 11-cat backbone P4， 第11层，Concat模块，将第10层和第6层的输出按通道拼接。
  - [-1, 3, C2f, [512]]  # 12， 第12层，C2f模块，重复3次，输出通道数为512，shortcut关闭。
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 13，第13层，上采样层，scale_factor=2，mode=nearest。
  - [[-1, 4], 1, Concat, [1]]  # 14-cat backbone P3， 第14层，Concat模块，将第13层和第4层的输出按通道拼接。
  - [-1, 3, C2f, [256]]  # 15-P3/8 (small) 第15层，C2f模块，重复3次，输出通道数为256，shortcut关闭。
  - [-1, 1, Conv, [256, 3, 2]]  # 16  第16层，Conv模块，步长为2。
  - [[-1, 12], 1, Concat, [1]]  # 17-cat head P4，第17层，Concat模块，将第16层和第12层的输出按通道拼接。
  - [-1, 3, C2f, [512]]  # 18-P4/16 (medium) ，第18层，C2f模块，重复3次，输出通道数为512，shortcut关闭。
  - [-1, 1, Conv, [512, 3, 2]]  # 19  第19层，Conv模块，步长为2。
  - [[-1, 9], 1, Concat, [1]]  # 20-cat head P5 ，第20层，Concat模块，将第19层和第9层的输出按通道拼接。
  - [-1, 3, C2f, [1024]]  # 21-P5/32 (large) ，第21层，C2f模块，重复3次，输出通道数为1024，shortcut关闭。
  - [[15, 18, 21], 1, Detect, [nc]]  # 22-Detect(P3, P4, P5) ，第22层，Detect模块，输入为第15层（P3）、第18层（P4）和第21层（P5）的输出。
"""
# YAML 内容作为多行字符串
yolov11x_config_yaml = """
# Optimized YOLOv11 object detection model with P3-P5 outputs. Targeting performance improvement without increasing parameters.

# Parameters
nc: 80  # number of classes
scales:  # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]  # YOLOv11n summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024]  # YOLOv11s summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
  m: [0.50, 1.00, 512]  # YOLOv11m summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [1.00, 1.00, 512]  # YOLOv11l summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [1.00, 1.50, 512]  # YOLOv11x summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLOv11n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2， 第0层，Conv模块，输入为原始图像(640x640x3)，卷积参数：输出通道数为64，卷积核大小为3x3，步长为2。
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4， 第1层，Conv模块，输入为上一层输出，输出通道数为128，卷积核大小为3x3，步长为2。
  - [-1, 2, C2f, [128, True]]  # 2， 第2层，C2f模块，重复2次，输出通道数为128，带有shortcut连接。
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8， 第3层，Conv模块，输入为上一层输出，输出通道数为256，卷积核大小为3x3，步长为2。
  - [-1, 4, C2f, [256, True]]  # 4，  第4层，C2f模块，重复4次，输出通道数为256，带有shortcut连接。
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16， 第5层，Conv模块，输入为上一层输出，输出通道数为512，卷积核大小为3x3，步长为2。
  - [-1, 6, C2f, [512, True]]  # 6 ， 第6层，C2f模块，重复6次，输出通道数为512，带有shortcut连接。
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32， 第7层，Conv模块，输入为上一层输出，输出通道数为1024，卷积核大小为3x3，步长为2。
  - [-1, 3, C2f, [1024, True]]  # 8， 第8层，C2f模块，重复3次，输出通道数为1024，带有shortcut连接。
  - [-1, 1, SPPF, [1024, 5]]  # 9， 第9层，SPPF模块（快速空间金字塔池化层），输出通道数为1024，池化核大小为5。

# YOLOv11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 10， 第10层，上采样层，输入为上一层输出，scale_factor=2，mode=nearest。
  - [[-1, 6], 1, Concat, [1]]  # 11-cat backbone P4， 第11层，Concat模块，将第10层和第6层的输出按通道拼接。
  - [-1, 3, C2f, [512]]  # 12， 第12层，C2f模块，重复3次，输出通道数为512，shortcut关闭。
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]  # 13，第13层，上采样层，scale_factor=2，mode=nearest。
  - [[-1, 4], 1, Concat, [1]]  # 14-cat backbone P3， 第14层，Concat模块，将第13层和第4层的输出按通道拼接。
  - [-1, 3, C2f, [256]]  # 15-P3/8 (small) 第15层，C2f模块，重复3次，输出通道数为256，shortcut关闭。
  - [-1, 1, Conv, [256, 3, 2]]  # 16  第16层，Conv模块，步长为2。
  - [[-1, 12], 1, Concat, [1]]  # 17-cat head P4，第17层，Concat模块，将第16层和第12层的输出按通道拼接。
  - [-1, 3, C2f, [512]]  # 18-P4/16 (medium) ，第18层，C2f模块，重复3次，输出通道数为512，shortcut关闭。
  - [-1, 1, Conv, [512, 3, 2]]  # 19  第19层，Conv模块，步长为2。
  - [[-1, 9], 1, Concat, [1]]  # 20-cat head P5 ，第20层，Concat模块，将第19层和第9层的输出按通道拼接。
  - [-1, 3, C2f, [1024]]  # 21-P5/32 (large) ，第21层，C2f模块，重复3次，输出通道数为1024，shortcut关闭。
  - [[15, 18, 21], 1, Detect, [nc]]  # 22-Detect(P3, P4, P5) ，第22层，Detect模块，输入为第15层（P3）、第18层（P4）和第21层（P5）的输出。
"""

modules = '''[ 'Bottleneck', 'C2f', 'C2fCIB', 'C3', 'C3k2', 'C3Ghost', 'Conv', 'GhostConv', 'SCDown', 'PSA', 'SPPF', 'C2PSA']
'''
modules_example = '''
Bottleneck: - [-1, 1, Bottleneck, [64]]
C2f: - [-1, 3, C2f, [128, True]]
C2fCIB: - [-1, 3, C2fCIB, [1024, True]]
C3: - [-1, 3, C3, [128]]
C3k2: - [-1, 2, C3k2, [1024, True]]
C3Ghost: - [-1, 6, C3Ghost, [256, True]]
Conv: - [-1, 1, Conv, [32, 3, 1]]
GhostConv: - [-1, 1, GhostConv, [128, 3, 2]]
PSA: - [-1, 1, PSA, [1024]]
SCDown: - [-1, 1, SCDown, [512, 3, 2]]
SPPF: - [-1, 1, SPPF, [1024, 5]]
C2PSA - [-1, 2, C2PSA, [1024]] # 10
'''


system_content = "You are Quoc V. Le, a computer scientist and artificial intelligence researcher who is widely regarded as one of the leading experts in deep learning and neural network architecture search. Your work in this area has focused on developing efficient algorithms for searching the space of possible neural network architectures, with the goal of finding architectures that perform well on a given task while minimizing the computational cost of training and inference."

# user_input = f'''You need to analyze where yolov11 is better than yolov11, and then understand and improve on the basis of yolov11 to make the newly generated configuration better than yolov11. The configuration file for yolov11 is {yolov11_config_yaml}, The configuration file for yolov11 is{yolov11_config_yaml}'''

'''更复杂一些的生成方式则是替换modul的类型，可选类型有：{modules}, 具体使用示例可参考{modules_example}。在替换module时必须注意的是module之间channel的匹配，特别要注意Concat这个过程，不要把channel匹配错了。如果你对某个modules不了解具体的结构，请不要使用。'''

higher_input = f'''A more complex way to generate modul is to replace the modul type with {modules}, for example {modules_example}. When replacing modules, you must pay attention to the matching of channels between modules. Pay special attention to the Concat process, and do not match the channels incorrectly. If you don't know the structure of modules, don't use them.'''

suffix = '''Please do not include anything else other than configuration in your response!'''


def generate_new_structure_using_llm_v11(scale, api_type, max_score_list, max_score_yaml_list, best_new_yaml_parameters_list, best_new_yaml_gflops_list, params, more_modules):
    
    # yolov11_config_yaml = get_yaml(scale)
    if scale == 'n':
        yolov11_config_yaml = yolov11n_config_yaml
    elif scale == 's':
        yolov11_config_yaml = yolov11s_config_yaml
    elif scale == 'm':
        yolov11_config_yaml = yolov11m_config_yaml
    elif scale == 'l':
        yolov11_config_yaml = yolov11l_config_yaml
    elif scale == 'x':
        yolov11_config_yaml = yolov11x_config_yaml
    else:
        raise ValueError(f"Invalid scale '{scale}'. Please choose from: 'n', 's', 'm', 'l', 'x'.")

    user_input = f'''You need to analyze yolov11 to make the newly generated configuration better than yolov11. The configuration file for yolov11 is {yolov11_config_yaml}
                 You can modify values in scales, repeats in backbone, channel in module, and channel in head. However, it is important to note that repeats no more than 10 times and the modified channel values need to match each other.
                 The methods to keep the number of parameters constant are as follows: 0. Only change the types of some modules, but the number of channels between modules must be strict; 1. Increase the number of layers while reducing the number of channels; 2. Increase the number of channels while reducing the number of layers. In short, the parameters, gradients and GFLOPs of the new configuration should not be increased. '''


    Parameters = params['parameters']
    GFLOPs =params['GFLOPs']
    Layers = params['layers']
    Gradients = params['gradients']
    
    # prompt_cn = '根据现有配置，生成一个新的优化后的配置，优化目标：参数量不增加或参数量减少情况下提升目标检测性能。具体的module顺序你可以更改，channel数值也可以变化。总之，你可以根据自己的理解生成新结构，结果比原有配置性能更优就可以。'
    prompt = f'''Generate a new optimized configuration based on the existing configuration.
             Optimization goal: Improve target detection performance when the number of parameters does not increase or decreases. 
             You can change the specific module order, and the channel value can also change. Sizes of tensors must match except in dimension 1.
             In short, you can generate a new structure according to your own understanding, and the result is better than the original configuration.'''
    
    Params_prompt = f"Please suggest a better structure that can improve the structure's score results provided above. The parameters of the new configuration should be less than {Parameters}, but more than half of {Parameters}. The GFLOPs of the new configuration should be less than {GFLOPs}."
        
    if len(max_score_list) > 0:
        experiments_prompt = lambda max_score_yaml_list, max_score_list, best_new_yaml_parameters_list, best_new_yaml_gflops_list : '''Here are some structure's score results that you can use as a reference:
        {}
        '''.format(''.join(['{} gives a score of {:.4f}, and {} parameters.\n\n'.format(best_structure, max_score, paramter, gflops) for best_structure, max_score, paramter, gflops in zip(max_score_yaml_list, max_score_list, best_new_yaml_parameters_list, best_new_yaml_gflops_list)]))
        if more_modules:    
            messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_input + higher_input + prompt + experiments_prompt(max_score_yaml_list, max_score_list, best_new_yaml_parameters_list, best_new_yaml_gflops_list) + Params_prompt + suffix}] 
        # print("# experiments_prompt: \n ", experiments_prompt(max_score_yaml_list, max_score_list))
        else:
            messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_input + prompt + experiments_prompt(max_score_yaml_list, max_score_list, best_new_yaml_parameters_list, best_new_yaml_gflops_list) + Params_prompt + suffix}] 
    else:
        
        if more_modules:
            messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_input + higher_input + prompt + Params_prompt + suffix}] 
        else:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_input + prompt + Params_prompt + suffix}] 

    if api_type == 'Poe':
        api_key = 'EUPOLRTFOHyyNRhjjEiRlRUMRjfFd0_bPZ2N4PE8PNE'
    elif api_type == 'gpt' :
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
        
    if api_type == 'Poe':
        # Create an asynchronous function to encapsulate the async for loop
        # ssh -L 9000:api.poe.com:443 feikong@172.18.20.193
        # ssh -N -R 9000:api.poe.com:443 kongfei@172.22.162.34
        # ssh -D 1080 feikong@172.18.20.193
        
        response_json = requests.post(
            'http://172.18.20.10:5001/get_responses',
            json={'api_key': api_key, 'messages': messages}
        )

        # print("# response:", response_json.json())
        response = response_json.json()['response']

    if api_type == 'gpt':
        # 配置 OpenAI API
        client = OpenAI(api_key=api_key, base_url="https://api.chatanywhere.tech/v1")
        response = client.chat.completions.create(
            model='gpt-4',  # 使用适当的模型名称
            # model='gpt-4o-mini',  # 
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
                # model="qwen2-72b-instruct", #  110（score）
                # model="qwen2.5-7b-instruct",  # 
                # model="qwen2.5-14b-instruct",  # 4
                # model="qwen2.5-32b-instruct",  # 80.43 / 20;  10 /min
                model="qwen2.5-72b-instruct",  #  99.60  60 / min    
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

    # 获取响应内容 
    if api_type != 'Poe':
        new_structure = response.choices[0].message.content
    else:
        new_structure = response

    # 去除 ```yaml 和 ```
    # cleaned_new_yaml = new_structure.strip("```yaml").strip("```")
    cleaned_new_yaml = clean_markdown_yaml(new_structure)


    return cleaned_new_yaml
