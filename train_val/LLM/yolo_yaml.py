
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


