from openai import OpenAI
from zhipuai import ZhipuAI
import time
import requests
from .yolo_yaml import clean_markdown_yaml


yolov8n_config_yaml = """
# Optimized YOLOv8 object detection model with P3-P5 outputs. Targeting performance improvement without increasing parameters.

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers, 3157200 parameters, 3157184 gradients, 8.9 GFLOPs

# YOLOv8n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2 第0层，-1代表将上层的输入作为本层的输入。第0层的输入是640*640*3的图像。Conv代表卷积层，相应的参数：64代表输出通道数，3代表卷积核大小k，2代表stride步长。
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4 第1层，本层和上一层是一样的操作（128代表输出通道数，3代表卷积核大小k，2代表stride步长）
  - [-1, 3, C2f, [128, True]] # 第2层，本层是C2f模块，3代表本层重复3次。128代表输出通道数，True表示Bottleneck有shortcut。
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8 第3层，进行卷积操作（256代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为80*80*256（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/8。
  - [-1, 6, C2f, [256, True]] # 第4层，本层是C2f模块，可以参考第2层的讲解。6代表本层重复6次。256代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是80*80*256。
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16 第5层，进行卷积操作（512代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为40*40*512（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/16。
  - [-1, 6, C2f, [512, True]] # 第6层，本层是C2f模块，可以参考第2层的讲解。6代表本层重复6次。512代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是40*40*512。
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32 第7层，进行卷积操作（1024代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为20*20*1024（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/32。
  - [-1, 3, C2f, [1024, True]] #第8层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。1024代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是20*20*1024。
  - [-1, 1, SPPF, [1024, 5]]  # 9 第9层，本层是快速空间金字塔池化层（SPPF）。1024代表输出通道数，5代表池化核大小k。结合模块结构图和代码可以看出，最后concat得到的特征图尺寸是20*20*（512*4），经过一次Conv得到20*20*1024。

# YOLOv8n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 第10层，本层是上采样层。-1代表将上层的输出作为本层的输入。None代表上采样的size=None（输出尺寸）不指定。2代表scale_factor=2，表示输出的尺寸是输入尺寸的2倍。mode=nearest代表使用的上采样算法为最近邻插值算法。经过这层之后，特征图的长和宽变成原来的两倍，通道数不变，所以最终尺寸为40*40*1024。
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4 第11层，本层是concat层，[-1, 6]代表将上层和第6层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是40*40*1024，第6层的输出是40*40*512，最终本层的输出尺寸为40*40*1536。
  - [-1, 3, C2f, [512]]  # 12 第12层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。512代表输出通道数。与Backbone中C2f不同的是，此处的C2f的bottleneck模块的shortcut=False。
 
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 第13层，本层也是上采样层（参考第10层）。经过这层之后，特征图的长和宽变成原来的两倍，通道数不变，所以最终尺寸为80*80*512。
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3 第14层，本层是concat层，[-1, 4]代表将上层和第4层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是80*80*512，第6层的输出是80*80*256，最终本层的输出尺寸为80*80*768。
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small) 第15层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。256代表输出通道数。经过这层之后，特征图尺寸变为80*80*256，特征图的长宽已经变成输入图像的1/8。
 
  - [-1, 1, Conv, [256, 3, 2]] # 第16层，进行卷积操作（256代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为40*40*256（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样）。
  - [[-1, 12], 1, Concat, [1]]  # cat head P4 第17层，本层是concat层，[-1, 12]代表将上层和第12层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是40*40*256，第12层的输出是40*40*512，最终本层的输出尺寸为40*40*768。
  - [-1, 3, C2f, [512]]  # 18 (P4/16-medium) 第18层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。512代表输出通道数。经过这层之后，特征图尺寸变为40*40*512，特征图的长宽已经变成输入图像的1/16。
 
  - [-1, 1, Conv, [512, 3, 2]] # 第19层，进行卷积操作（512代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为20*20*512（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样）。
  - [[-1, 9], 1, Concat, [1]]  # cat head P5 第20层，本层是concat层，[-1, 9]代表将上层和第9层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是20*20*512，第9层的输出是20*20*1024，最终本层的输出尺寸为20*20*1536。
  - [-1, 3, C2f, [1024]]  # 21 (P5/32-large) 第21层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。1024代表输出通道数。经过这层之后，特征图尺寸变为20*20*1024，特征图的长宽已经变成输入图像的1/32。
 
  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5) 第20层，本层是Detect层，[15, 18, 21]代表将第15、18、21层的输出（分别是80*80*256、40*40*512、20*20*1024）作为本层的输入。nc是数据集的类别数。
"""
# YAML 内容作为多行字符串
yolov8s_config_yaml = """
# Optimized YOLOv8 object detection model with P3-P5 outputs. Targeting performance improvement without increasing parameters.

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs

# YOLOv8n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2 第0层，-1代表将上层的输入作为本层的输入。第0层的输入是640*640*3的图像。Conv代表卷积层，相应的参数：64代表输出通道数，3代表卷积核大小k，2代表stride步长。
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4 第1层，本层和上一层是一样的操作（128代表输出通道数，3代表卷积核大小k，2代表stride步长）
  - [-1, 3, C2f, [128, True]] # 第2层，本层是C2f模块，3代表本层重复3次。128代表输出通道数，True表示Bottleneck有shortcut。
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8 第3层，进行卷积操作（256代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为80*80*256（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/8。
  - [-1, 6, C2f, [256, True]] # 第4层，本层是C2f模块，可以参考第2层的讲解。6代表本层重复6次。256代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是80*80*256。
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16 第5层，进行卷积操作（512代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为40*40*512（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/16。
  - [-1, 6, C2f, [512, True]] # 第6层，本层是C2f模块，可以参考第2层的讲解。6代表本层重复6次。512代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是40*40*512。
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32 第7层，进行卷积操作（1024代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为20*20*1024（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/32。
  - [-1, 3, C2f, [1024, True]] #第8层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。1024代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是20*20*1024。
  - [-1, 1, SPPF, [1024, 5]]  # 9 第9层，本层是快速空间金字塔池化层（SPPF）。1024代表输出通道数，5代表池化核大小k。结合模块结构图和代码可以看出，最后concat得到的特征图尺寸是20*20*（512*4），经过一次Conv得到20*20*1024。

# YOLOv8n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 第10层，本层是上采样层。-1代表将上层的输出作为本层的输入。None代表上采样的size=None（输出尺寸）不指定。2代表scale_factor=2，表示输出的尺寸是输入尺寸的2倍。mode=nearest代表使用的上采样算法为最近邻插值算法。经过这层之后，特征图的长和宽变成原来的两倍，通道数不变，所以最终尺寸为40*40*1024。
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4 第11层，本层是concat层，[-1, 6]代表将上层和第6层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是40*40*1024，第6层的输出是40*40*512，最终本层的输出尺寸为40*40*1536。
  - [-1, 3, C2f, [512]]  # 12 第12层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。512代表输出通道数。与Backbone中C2f不同的是，此处的C2f的bottleneck模块的shortcut=False。
 
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 第13层，本层也是上采样层（参考第10层）。经过这层之后，特征图的长和宽变成原来的两倍，通道数不变，所以最终尺寸为80*80*512。
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3 第14层，本层是concat层，[-1, 4]代表将上层和第4层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是80*80*512，第6层的输出是80*80*256，最终本层的输出尺寸为80*80*768。
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small) 第15层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。256代表输出通道数。经过这层之后，特征图尺寸变为80*80*256，特征图的长宽已经变成输入图像的1/8。
 
  - [-1, 1, Conv, [256, 3, 2]] # 第16层，进行卷积操作（256代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为40*40*256（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样）。
  - [[-1, 12], 1, Concat, [1]]  # cat head P4 第17层，本层是concat层，[-1, 12]代表将上层和第12层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是40*40*256，第12层的输出是40*40*512，最终本层的输出尺寸为40*40*768。
  - [-1, 3, C2f, [512]]  # 18 (P4/16-medium) 第18层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。512代表输出通道数。经过这层之后，特征图尺寸变为40*40*512，特征图的长宽已经变成输入图像的1/16。
 
  - [-1, 1, Conv, [512, 3, 2]] # 第19层，进行卷积操作（512代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为20*20*512（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样）。
  - [[-1, 9], 1, Concat, [1]]  # cat head P5 第20层，本层是concat层，[-1, 9]代表将上层和第9层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是20*20*512，第9层的输出是20*20*1024，最终本层的输出尺寸为20*20*1536。
  - [-1, 3, C2f, [1024]]  # 21 (P5/32-large) 第21层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。1024代表输出通道数。经过这层之后，特征图尺寸变为20*20*1024，特征图的长宽已经变成输入图像的1/32。
 
  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5) 第20层，本层是Detect层，[15, 18, 21]代表将第15、18、21层的输出（分别是80*80*256、40*40*512、20*20*1024）作为本层的输入。nc是数据集的类别数。
"""
# YAML 内容作为多行字符串
yolov8m_config_yaml = """
# Optimized YOLOv8 object detection model with P3-P5 outputs. Targeting performance improvement without increasing parameters.

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs

# YOLOv8n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2 第0层，-1代表将上层的输入作为本层的输入。第0层的输入是640*640*3的图像。Conv代表卷积层，相应的参数：64代表输出通道数，3代表卷积核大小k，2代表stride步长。
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4 第1层，本层和上一层是一样的操作（128代表输出通道数，3代表卷积核大小k，2代表stride步长）
  - [-1, 3, C2f, [128, True]] # 第2层，本层是C2f模块，3代表本层重复3次。128代表输出通道数，True表示Bottleneck有shortcut。
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8 第3层，进行卷积操作（256代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为80*80*256（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/8。
  - [-1, 6, C2f, [256, True]] # 第4层，本层是C2f模块，可以参考第2层的讲解。6代表本层重复6次。256代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是80*80*256。
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16 第5层，进行卷积操作（512代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为40*40*512（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/16。
  - [-1, 6, C2f, [512, True]] # 第6层，本层是C2f模块，可以参考第2层的讲解。6代表本层重复6次。512代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是40*40*512。
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32 第7层，进行卷积操作（1024代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为20*20*1024（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/32。
  - [-1, 3, C2f, [1024, True]] #第8层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。1024代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是20*20*1024。
  - [-1, 1, SPPF, [1024, 5]]  # 9 第9层，本层是快速空间金字塔池化层（SPPF）。1024代表输出通道数，5代表池化核大小k。结合模块结构图和代码可以看出，最后concat得到的特征图尺寸是20*20*（512*4），经过一次Conv得到20*20*1024。

# YOLOv8n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 第10层，本层是上采样层。-1代表将上层的输出作为本层的输入。None代表上采样的size=None（输出尺寸）不指定。2代表scale_factor=2，表示输出的尺寸是输入尺寸的2倍。mode=nearest代表使用的上采样算法为最近邻插值算法。经过这层之后，特征图的长和宽变成原来的两倍，通道数不变，所以最终尺寸为40*40*1024。
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4 第11层，本层是concat层，[-1, 6]代表将上层和第6层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是40*40*1024，第6层的输出是40*40*512，最终本层的输出尺寸为40*40*1536。
  - [-1, 3, C2f, [512]]  # 12 第12层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。512代表输出通道数。与Backbone中C2f不同的是，此处的C2f的bottleneck模块的shortcut=False。
 
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 第13层，本层也是上采样层（参考第10层）。经过这层之后，特征图的长和宽变成原来的两倍，通道数不变，所以最终尺寸为80*80*512。
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3 第14层，本层是concat层，[-1, 4]代表将上层和第4层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是80*80*512，第6层的输出是80*80*256，最终本层的输出尺寸为80*80*768。
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small) 第15层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。256代表输出通道数。经过这层之后，特征图尺寸变为80*80*256，特征图的长宽已经变成输入图像的1/8。
 
  - [-1, 1, Conv, [256, 3, 2]] # 第16层，进行卷积操作（256代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为40*40*256（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样）。
  - [[-1, 12], 1, Concat, [1]]  # cat head P4 第17层，本层是concat层，[-1, 12]代表将上层和第12层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是40*40*256，第12层的输出是40*40*512，最终本层的输出尺寸为40*40*768。
  - [-1, 3, C2f, [512]]  # 18 (P4/16-medium) 第18层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。512代表输出通道数。经过这层之后，特征图尺寸变为40*40*512，特征图的长宽已经变成输入图像的1/16。
 
  - [-1, 1, Conv, [512, 3, 2]] # 第19层，进行卷积操作（512代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为20*20*512（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样）。
  - [[-1, 9], 1, Concat, [1]]  # cat head P5 第20层，本层是concat层，[-1, 9]代表将上层和第9层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是20*20*512，第9层的输出是20*20*1024，最终本层的输出尺寸为20*20*1536。
  - [-1, 3, C2f, [1024]]  # 21 (P5/32-large) 第21层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。1024代表输出通道数。经过这层之后，特征图尺寸变为20*20*1024，特征图的长宽已经变成输入图像的1/32。
 
  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5) 第20层，本层是Detect层，[15, 18, 21]代表将第15、18、21层的输出（分别是80*80*256、40*40*512、20*20*1024）作为本层的输入。nc是数据集的类别数。
"""
# YAML 内容作为多行字符串
yolov8l_config_yaml = """
# Optimized YOLOv8 object detection model with P3-P5 outputs. Targeting performance improvement without increasing parameters.

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs

# YOLOv8n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2 第0层，-1代表将上层的输入作为本层的输入。第0层的输入是640*640*3的图像。Conv代表卷积层，相应的参数：64代表输出通道数，3代表卷积核大小k，2代表stride步长。
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4 第1层，本层和上一层是一样的操作（128代表输出通道数，3代表卷积核大小k，2代表stride步长）
  - [-1, 3, C2f, [128, True]] # 第2层，本层是C2f模块，3代表本层重复3次。128代表输出通道数，True表示Bottleneck有shortcut。
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8 第3层，进行卷积操作（256代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为80*80*256（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/8。
  - [-1, 6, C2f, [256, True]] # 第4层，本层是C2f模块，可以参考第2层的讲解。6代表本层重复6次。256代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是80*80*256。
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16 第5层，进行卷积操作（512代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为40*40*512（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/16。
  - [-1, 6, C2f, [512, True]] # 第6层，本层是C2f模块，可以参考第2层的讲解。6代表本层重复6次。512代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是40*40*512。
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32 第7层，进行卷积操作（1024代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为20*20*1024（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/32。
  - [-1, 3, C2f, [1024, True]] #第8层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。1024代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是20*20*1024。
  - [-1, 1, SPPF, [1024, 5]]  # 9 第9层，本层是快速空间金字塔池化层（SPPF）。1024代表输出通道数，5代表池化核大小k。结合模块结构图和代码可以看出，最后concat得到的特征图尺寸是20*20*（512*4），经过一次Conv得到20*20*1024。

# YOLOv8n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 第10层，本层是上采样层。-1代表将上层的输出作为本层的输入。None代表上采样的size=None（输出尺寸）不指定。2代表scale_factor=2，表示输出的尺寸是输入尺寸的2倍。mode=nearest代表使用的上采样算法为最近邻插值算法。经过这层之后，特征图的长和宽变成原来的两倍，通道数不变，所以最终尺寸为40*40*1024。
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4 第11层，本层是concat层，[-1, 6]代表将上层和第6层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是40*40*1024，第6层的输出是40*40*512，最终本层的输出尺寸为40*40*1536。
  - [-1, 3, C2f, [512]]  # 12 第12层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。512代表输出通道数。与Backbone中C2f不同的是，此处的C2f的bottleneck模块的shortcut=False。
 
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 第13层，本层也是上采样层（参考第10层）。经过这层之后，特征图的长和宽变成原来的两倍，通道数不变，所以最终尺寸为80*80*512。
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3 第14层，本层是concat层，[-1, 4]代表将上层和第4层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是80*80*512，第6层的输出是80*80*256，最终本层的输出尺寸为80*80*768。
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small) 第15层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。256代表输出通道数。经过这层之后，特征图尺寸变为80*80*256，特征图的长宽已经变成输入图像的1/8。
 
  - [-1, 1, Conv, [256, 3, 2]] # 第16层，进行卷积操作（256代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为40*40*256（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样）。
  - [[-1, 12], 1, Concat, [1]]  # cat head P4 第17层，本层是concat层，[-1, 12]代表将上层和第12层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是40*40*256，第12层的输出是40*40*512，最终本层的输出尺寸为40*40*768。
  - [-1, 3, C2f, [512]]  # 18 (P4/16-medium) 第18层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。512代表输出通道数。经过这层之后，特征图尺寸变为40*40*512，特征图的长宽已经变成输入图像的1/16。
 
  - [-1, 1, Conv, [512, 3, 2]] # 第19层，进行卷积操作（512代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为20*20*512（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样）。
  - [[-1, 9], 1, Concat, [1]]  # cat head P5 第20层，本层是concat层，[-1, 9]代表将上层和第9层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是20*20*512，第9层的输出是20*20*1024，最终本层的输出尺寸为20*20*1536。
  - [-1, 3, C2f, [1024]]  # 21 (P5/32-large) 第21层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。1024代表输出通道数。经过这层之后，特征图尺寸变为20*20*1024，特征图的长宽已经变成输入图像的1/32。
 
  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5) 第20层，本层是Detect层，[15, 18, 21]代表将第15、18、21层的输出（分别是80*80*256、40*40*512、20*20*1024）作为本层的输入。nc是数据集的类别数。
"""
# YAML 内容作为多行字符串
yolov8x_config_yaml = """
# Optimized YOLOv8 object detection model with P3-P5 outputs. Targeting performance improvement without increasing parameters.

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs

# YOLOv8n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2 第0层，-1代表将上层的输入作为本层的输入。第0层的输入是640*640*3的图像。Conv代表卷积层，相应的参数：64代表输出通道数，3代表卷积核大小k，2代表stride步长。
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4 第1层，本层和上一层是一样的操作（128代表输出通道数，3代表卷积核大小k，2代表stride步长）
  - [-1, 3, C2f, [128, True]] # 第2层，本层是C2f模块，3代表本层重复3次。128代表输出通道数，True表示Bottleneck有shortcut。
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8 第3层，进行卷积操作（256代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为80*80*256（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/8。
  - [-1, 6, C2f, [256, True]] # 第4层，本层是C2f模块，可以参考第2层的讲解。6代表本层重复6次。256代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是80*80*256。
  - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16 第5层，进行卷积操作（512代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为40*40*512（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/16。
  - [-1, 6, C2f, [512, True]] # 第6层，本层是C2f模块，可以参考第2层的讲解。6代表本层重复6次。512代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是40*40*512。
  - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32 第7层，进行卷积操作（1024代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为20*20*1024（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样），特征图的长宽已经变成输入图像的1/32。
  - [-1, 3, C2f, [1024, True]] #第8层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。1024代表输出通道数，True表示Bottleneck有shortcut。经过这层之后，特征图尺寸依旧是20*20*1024。
  - [-1, 1, SPPF, [1024, 5]]  # 9 第9层，本层是快速空间金字塔池化层（SPPF）。1024代表输出通道数，5代表池化核大小k。结合模块结构图和代码可以看出，最后concat得到的特征图尺寸是20*20*（512*4），经过一次Conv得到20*20*1024。

# YOLOv8n head
head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 第10层，本层是上采样层。-1代表将上层的输出作为本层的输入。None代表上采样的size=None（输出尺寸）不指定。2代表scale_factor=2，表示输出的尺寸是输入尺寸的2倍。mode=nearest代表使用的上采样算法为最近邻插值算法。经过这层之后，特征图的长和宽变成原来的两倍，通道数不变，所以最终尺寸为40*40*1024。
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4 第11层，本层是concat层，[-1, 6]代表将上层和第6层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是40*40*1024，第6层的输出是40*40*512，最终本层的输出尺寸为40*40*1536。
  - [-1, 3, C2f, [512]]  # 12 第12层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。512代表输出通道数。与Backbone中C2f不同的是，此处的C2f的bottleneck模块的shortcut=False。
 
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']] # 第13层，本层也是上采样层（参考第10层）。经过这层之后，特征图的长和宽变成原来的两倍，通道数不变，所以最终尺寸为80*80*512。
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3 第14层，本层是concat层，[-1, 4]代表将上层和第4层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是80*80*512，第6层的输出是80*80*256，最终本层的输出尺寸为80*80*768。
  - [-1, 3, C2f, [256]]  # 15 (P3/8-small) 第15层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。256代表输出通道数。经过这层之后，特征图尺寸变为80*80*256，特征图的长宽已经变成输入图像的1/8。
 
  - [-1, 1, Conv, [256, 3, 2]] # 第16层，进行卷积操作（256代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为40*40*256（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样）。
  - [[-1, 12], 1, Concat, [1]]  # cat head P4 第17层，本层是concat层，[-1, 12]代表将上层和第12层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是40*40*256，第12层的输出是40*40*512，最终本层的输出尺寸为40*40*768。
  - [-1, 3, C2f, [512]]  # 18 (P4/16-medium) 第18层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。512代表输出通道数。经过这层之后，特征图尺寸变为40*40*512，特征图的长宽已经变成输入图像的1/16。
 
  - [-1, 1, Conv, [512, 3, 2]] # 第19层，进行卷积操作（512代表输出通道数，3代表卷积核大小k，2代表stride步长），输出特征图尺寸为20*20*512（卷积的参数都没变，所以都是长宽变成原来的1/2，和之前一样）。
  - [[-1, 9], 1, Concat, [1]]  # cat head P5 第20层，本层是concat层，[-1, 9]代表将上层和第9层的输出作为本层的输入。[1]代表concat拼接的维度是1。从上面的分析可知，上层的输出尺寸是20*20*512，第9层的输出是20*20*1024，最终本层的输出尺寸为20*20*1536。
  - [-1, 3, C2f, [1024]]  # 21 (P5/32-large) 第21层，本层是C2f模块，可以参考第2层的讲解。3代表本层重复3次。1024代表输出通道数。经过这层之后，特征图尺寸变为20*20*1024，特征图的长宽已经变成输入图像的1/32。
 
  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5) 第20层，本层是Detect层，[15, 18, 21]代表将第15、18、21层的输出（分别是80*80*256、40*40*512、20*20*1024）作为本层的输入。nc是数据集的类别数。
"""


modules = '''[ 'Bottleneck', 'C2f', 'C2fCIB', 'C3', 'C3Ghost', 'Conv', 'GhostConv', 'SCDown', 'PSA', 'SPPF']
'''
modules_example = '''
Bottleneck: - [-1, 1, Bottleneck, [64]]
C2f: - [-1, 3, C2f, [128, True]]
C2fCIB: - [-1, 3, C2fCIB, [1024, True]]
C3: - [-1, 3, C3, [128]]
C3Ghost: - [-1, 6, C3Ghost, [256, True]]
Conv: - [-1, 1, Conv, [32, 3, 1]]
GhostConv: - [-1, 1, GhostConv, [128, 3, 2]]
PSA: - [-1, 1, PSA, [1024]]
SCDown: - [-1, 1, SCDown, [512, 3, 2]]
SPPF: - [-1, 1, SPPF, [1024, 5]]
'''


system_content = "You are Quoc V. Le, a computer scientist and artificial intelligence researcher who is widely regarded as one of the leading experts in deep learning and neural network architecture search. Your work in this area has focused on developing efficient algorithms for searching the space of possible neural network architectures, with the goal of finding architectures that perform well on a given task while minimizing the computational cost of training and inference."

# user_input = f'''You need to analyze where yolov11 is better than yolov8, and then understand and improve on the basis of yolov8 to make the newly generated configuration better than yolov8. The configuration file for yolov8 is {yolov8_config_yaml}, The configuration file for yolov11 is{yolov11_config_yaml}'''

'''更复杂一些的生成方式则是替换modul的类型，可选类型有：{modules}, 具体使用示例可参考{modules_example}。在替换module时必须注意的是module之间channel的匹配，特别要注意Concat这个过程，不要把channel匹配错了。如果你对某个modules不了解具体的结构，请不要使用。'''

higher_input = f'''A more complex way to generate modul is to replace the modul type with {modules}, for example {modules_example}. When replacing modules, you must pay attention to the matching of channels between modules. Pay special attention to the Concat process, and do not match the channels incorrectly. If you don't know the structure of modules, don't use them.'''

suffix = '''Please do not include anything else other than configuration in your response!'''


def generate_new_structure_using_llm(scale, api_type, max_score_list, max_score_yaml_list, best_new_yaml_parameters_list, best_new_yaml_gflops_list, params, more_modules):
    
    # yolov8_config_yaml = get_yaml(scale)
    if scale == 'n':
        yolov8_config_yaml = yolov8n_config_yaml
    elif scale == 's':
        yolov8_config_yaml = yolov8s_config_yaml
    elif scale == 'm':
        yolov8_config_yaml = yolov8m_config_yaml
    elif scale == 'l':
        yolov8_config_yaml = yolov8l_config_yaml
    elif scale == 'x':
        yolov8_config_yaml = yolov8x_config_yaml
    else:
        raise ValueError(f"Invalid scale '{scale}'. Please choose from: 'n', 's', 'm', 'l', 'x'.")

    user_input = f'''You need to analyze yolov8 to make the newly generated configuration better than yolov8. The configuration file for yolov8 is {yolov8_config_yaml}
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
