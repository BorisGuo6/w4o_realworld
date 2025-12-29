import os
import re
from typing import Optional

def get_newest(parent_dir: str) -> Optional[str]:
    """
    扫描一个目录，查找所有以 'YYYYMMDD_HHMMSS' 格式命名的子文件夹，
    并返回代表最新日期时间的那个文件夹的名称。

    Args:
        parent_dir: 要扫描的父目录的路径。

    Returns:
        一个字符串，表示最新的文件夹名称。
        如果找不到匹配的文件夹或父目录不存在，则返回 None。
    """
    # 步骤1: 验证父目录是否存在
    if not os.path.isdir(parent_dir):
        print(f"错误: 目录 '{parent_dir}' 不存在。")
        return None

    # 步骤2: 定义用于验证文件夹名称格式的正则表达式
    # ^      -> 字符串开头
    # \d{8}  -> 恰好8个数字 (对应 YYYYMMDD)
    # _      -> 一个下划线
    # \d{6}  -> 恰好6个数字 (对应 HHMMSS)
    # $      -> 字符串结尾
    pattern = re.compile(r'^\d{8}_\d{6}$')

    # 步骤3: 筛选出所有有效的子文件夹名称
    valid_folders = []
    for item_name in os.listdir(parent_dir):
        # 构造完整的路径
        full_path = os.path.join(parent_dir, item_name)
        # 检查它是否是一个文件夹，并且名称是否符合我们的格式
        if os.path.isdir(full_path) and pattern.match(item_name):
            valid_folders.append(item_name)
    
    # 步骤4: 找到最新的文件夹
    # 如果列表不为空，直接使用 max() 函数找到字符串最大的那个
    if valid_folders:
        latest_folder = max(valid_folders)
        return latest_folder
    else:
        # 如果列表为空，说明没有找到任何匹配的文件夹
        return None

# --- 使用示例 ---
if __name__ == "__main__":
    # 假设您的项目结构如下:
    # /path/to/your/data/
    # ├── 20250902_183000
    # ├── 20250903_170052
    # ├── 20250903_193010  <-- 这是最新的
    # ├── some_other_folder
    # └── a_file.txt

    # 为了演示，我们先手动创建一些示例文件夹和文件
    base_path = "temp_data_demo"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    folders_to_create = [
        "20250902_183000",
        "20250903_170052",
        "20250903_193010", # 最新的
        "not_a_valid_format",
        "20250903_193010_extra" # 格式不匹配
    ]
    for folder in folders_to_create:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)
    # 创建一个文件来确保函数能正确忽略它
    with open(os.path.join(base_path, "log.txt"), "w") as f:
        f.write("hello")
        
    print(f"在 '{base_path}' 文件夹中查找...")

    # --- 调用函数 ---
    latest_folder_name = get_newest(base_path)

    # --- 输出结果 ---
    if latest_folder_name:
        print(f"找到的最新文件夹是: {latest_folder_name}")
        # 您可以接着构造完整的路径来使用
        latest_folder_path = os.path.join(base_path, latest_folder_name)
        print(f"其完整路径是: {latest_folder_path}")
    else:
        print("没有找到任何符合 'YYYYMMDD_HHMMSS' 格式的子文件夹。")

    print("\n--- 测试一个不存在的路径 ---")
    find_latest_capture_folder("non_existent_directory")