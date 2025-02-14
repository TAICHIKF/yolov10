
import re

def clean_markdown_yaml(raw_text):
        """
        清理带有 Markdown 标记的 YAML 文本
        """
        # 定义正则表达式，提取 ```yaml ... ``` 或 ``` ... ``` 中的内容
        pattern = r"```(?:yaml)?\n(.*?)```"  # 支持 `yaml` 和无标签的 Markdown
        match = re.search(pattern, raw_text, re.DOTALL)  # DOTALL 允许匹配多行内容
        if match:
            # 返回提取的 YAML 内容，并移除首尾空白符
            return match.group(1).strip()
        else:
            raise ValueError("No valid YAML content found in the input text")


def clean_markdown_yaml_ds(raw_text):
    """
    清理带有 Markdown 标记的 YAML 文本，并去除 <think> 标签
    """
    # 去除 <think> 标签及其中的内容
    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)

    # 返回清理后的内容，去除首尾空白符
    return raw_text.strip()
    

def clean_and_extract_optimized_config(raw_text, comment_after_detect=True):
    """
    清理带有 Markdown 标记的 YAML 文本，去除 <think> 标签，并提取优化后的 YOLO 配置文本。
    可以选择将 `Detect, [nc]]` 之后的内容注释掉或者删除。
    """
    # 去除 <think> 标签及其中的内容
    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)

    # 提取配置内容，从 `nc: 80` 到 `Detect` 部分
    pattern = r"(nc: 80.*?)(?=\[\[15, 18, 21], 1, Detect, \[nc\]\])"
    match = re.search(pattern, raw_text, re.DOTALL)

    if match:
        # 提取配置内容
        config_content = match.group(1)

        # 找到 `- [[15, 18, 21], 1, Detect, [nc]]` 并保留该行及其之前的部分
        detect_line = r"- \[\[15, 18, 21], 1, Detect, \[nc\]\]"

        # 保留 `- [[15, 18, 21], 1, Detect, [nc]]` 及其之前的所有内容，删除之后的部分
        pattern_after_detect = r"(" + re.escape(detect_line) + r".*)"
        match_after_detect = re.search(pattern_after_detect, raw_text, re.DOTALL)
        
        if match_after_detect:
            config_content = config_content + "\n" + match_after_detect.group(1).split('\n')[0]  # 只保留第一行

        return config_content.strip()
    else:
        raise ValueError("No valid configuration content found.")