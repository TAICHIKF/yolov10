import pandas as pd
import os

def get_max(file_path):
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在！")
        return
    
    data = pd.read_csv(file_path)
    # 去除列名前后的空格
    data.columns = data.columns.str.strip()
    
    print("列名：", repr(data.columns))  # 显示列名

    # 获取 `metrics/mAP50-95(B)` 的最大值及对应的 epoch
    column_name = "metrics/mAP50-95(B)"  # 确保列名正确
    if column_name in data.columns:
        print("数据头部：", data.head())  # 查看前几行数据
        print("数据类型：", data[column_name].dtype)  # 查看列数据类型
        
        # 检查是否有NaN值并去除
        if data[column_name].isna().sum() > 0:
            print(f"警告：'{column_name}' 列有 {data[column_name].isna().sum()} 个缺失值，正在去除...")
            data = data.dropna(subset=[column_name])
        
        max_value = data[column_name].max()  # 最大值
        max_epoch = data.loc[data[column_name].idxmax(), "epoch"]  # 对应的 epoch
        print(f"最大 mAP50-95(B): {max_value}, 对应的 epoch: {max_epoch}")
    else:
        print(f"列 '{column_name}' 不存在，请检查列名是否正确。")

# get_max('/home/kongfei/code/yolov10/runs/detect/yolov8puls10/results.csv')
get_max('/home/kongfei/code/yolov10/llmnas_yolov8/Poe_20_l_yolov8plus17l/results.csv')
