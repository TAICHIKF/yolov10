from pathlib import Path

# 假设 args 是一个包含训练参数的对象
class Args:
    def __init__(self, save_dir=None, name=None, mode='train', exist_ok=False):
        self.save_dir = save_dir
        self.name = name
        self.mode = mode
        self.exist_ok = exist_ok


# 定义 get_save_dir 函数
def get_save_dir(args, name=None):
    """Return save_dir as created from train/val/predict arguments."""

    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path

        project = "1"  # 定义项目目录
        name = name or args.name or f"{args.mode}"  # 根据模式选择名称
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok)

    return Path(save_dir)

# 假设我们传递给 args 的参数
args = Args(save_dir="./runs/detect/llmnas/yolov8plus10",name="my_model", mode="train", exist_ok=True)

# 获取保存路径
save_dir = get_save_dir(args)

# 打印保存路径
print(f"模型将保存到: {save_dir}")