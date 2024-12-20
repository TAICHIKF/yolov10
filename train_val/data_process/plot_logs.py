import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取 CSV 文件
csv_dir = '/home/kongfei/code/yolov10/runs/detect/train_10b'

csv_file_path = 'results.csv'  # 替换为你的 CSV 文件路径

# 检查路径是否正确
if os.path.exists(os.path.join(csv_dir, csv_file_path)):

    # 假设CSV文件的列名如下（注意：这里需要根据你的实际数据调整列名列表）  
    correct_column_names = [  
        'time', 'epoch', 'train/box_om', 'train/cls_om', 'train/dfl_om',  
        'train/box_oo', 'train/cls_oo', 'train/dfl_oo', 'metrics/precision(B)',  
        'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',  
        'val/box_om', 'val/cls_om', 'val/dfl_om', 'val/box_oo', 'val/cls_oo',  
        'val/dfl_oo', 'lr/pg0', 'lr/pg1', 'lr/pg2'  
    ]  

    # 读取CSV文件，跳过第一行（错误的列名行），并指定正确的列名  
    df = pd.read_csv(os.path.join(csv_dir, csv_file_path), skiprows=1, names=correct_column_names, index_col=False)  
    
    print("文件读取成功！")
    
    # 打印原始列名以确认
    print("原始列名：", df.columns)
    print(df.head())

    # 获取 metrics/mAP50-95(B) 的最大值及对应的 epoch
    max_map_value = df['metrics/mAP50-95(B)'].max()
    max_map_epoch = df['epoch'][df['metrics/mAP50-95(B)'].idxmax()]

    # 打印最大值及对应的 epoch
    print(f"metrics/mAP50-95(B) 最大值: {max_map_value} (对应 epoch: {max_map_epoch})")

else:
    print(f"文件不存在，请检查路径：{os.path.join(csv_dir, csv_file_path)}")

# 设置绘图风格
sns.set(style="darkgrid")

# 定义需要绘制的列
columns_to_plot = [
    'train/cls_om', 'train/dfl_om',
    'train/box_oo', 'train/cls_oo', 'train/dfl_oo',
    'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
    'val/box_om', 'val/cls_om', 'val/dfl_om',
    'val/box_oo', 'val/cls_oo', 'val/dfl_oo',
    'lr/pg0', 'lr/pg1', 'lr/pg2',
    # 'train/box_om',
]

# 创建一个大的图形区域，以便放置多个子图
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(50, 40))  # 调整了图像大小以适应常规显示
axes = axes.flatten()

# 绘制每个指标
for i, column in enumerate(columns_to_plot):
    ax = axes[i]
    
    # 使用 epoch 作为 x 轴
    sns.lineplot(x='epoch', y=column, data=df, ax=ax)
    ax.set_title(column)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(column)

    # 确保 x 轴刻度是整数
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# 如果有剩余的子图区域，隐藏它们
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# 保存绘制的图像到文件
output_image_path = os.path.join(csv_dir, 'training_metrics_plot.png')  # 指定输出图片路径
plt.savefig(output_image_path, dpi=300)  # 保存图像，dpi设置图片质量
print(f"图像已保存至：{output_image_path}")

# 调整布局
plt.tight_layout()
plt.show()