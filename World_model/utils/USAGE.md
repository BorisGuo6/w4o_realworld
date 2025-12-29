# World4Omni 使用说明

## 项目概述

World4Omni 是一个集成了图像编辑、目标检测、分割和验证的完整AI系统。它结合了Google GenAI、Grounded SAM 2和迭代验证技术，提供高质量的图像编辑和物体操作能力。

## 核心功能

### 1. World Model (utils/World_model.py)

**功能**: 完整的图像编辑管道，支持多种模式配置

**基本用法**:
```python
from utils.World_model import World_model

# 完整模式（推荐）
result_path = World_model(
    image="images/move_tomato_to_pan.png",
    text="move the tomato to pan"
)
```

**高级配置**:
```python
# 自定义配置
result_path = World_model(
    image="images/move_tomato_to_pan.png",
    text="move the tomato to pan",
    generate_masks=True,           # 生成掩码
    cleanup_intermediate=True,     # 清理中间文件
    box_threshold=0.35,           # 检测阈值
    text_threshold=0.25,          # 文本阈值
    max_iterations=3,             # 最大迭代次数
    use_enhancer=True,            # 使用指令增强
    use_reflector=True            # 使用验证和反思
)
```

**参数说明**:
- `image`: 输入图像路径
- `text`: 文字指令
- `generate_masks`: 是否生成目标物体掩码（默认: True）
- `cleanup_intermediate`: 是否清理中间文件（默认: True）
- `box_threshold`: GroundingDINO 边界框阈值（默认: 0.35）
- `text_threshold`: GroundingDINO 文本阈值（默认: 0.25）
- `max_iterations`: 最大验证迭代次数（默认: 3）
- `use_enhancer`: 是否使用指令增强（默认: True）
- `use_reflector`: 是否使用验证和反思（默认: True）

**工作流程**:
1. **指令增强**: 将自然语言转换为精确的编辑指令
2. **迭代合成**: 运行图像编辑并进行验证循环
3. **Gemini验证**: AI验证结果是否满足要求
4. **反思改进**: 如果验证失败，生成修订指令
5. **掩码生成**: 为原始图像和编辑图像创建物体掩码
6. **清理**: 删除中间文件，保留最终结果

**输出结构**:
```
outputs/mask/run_YYYYMMDD_HHMMSS/
├── original.png          # 原始图像
├── edited.png            # 编辑图像
├── original_mask.png     # 原始掩码
└── edited_mask.png       # 编辑掩码
```

## 脚本使用

### 测试脚本

**基本测试**:
```bash
export GEMINI_API_KEY=YOUR_KEY
python scripts/test_world_model.py
```

**多模式测试**:
```bash
python scripts/test_world_model_modes.py
```

### 独立脚本

**图像生成**:
```bash
python scripts/gen_img.py "a red circle on white background"
```

**图像编辑**:
```bash
python scripts/edit_img.py image.png "edit instruction"
```

**指令增强**:
```bash
python scripts/enhancer.py "move the tomato to pan"
```

**物体提取**:
```bash
python scripts/extract_objects.py "move the tomato to pan"
```

**Grounded SAM**:
```bash
python scripts/test_grounded_sam.py image.png "tomato, pan"
```

**完整反射器**:
```bash
python scripts/Reflector.py image.png "instruction"
```

## 环境配置

### 必需环境变量
```bash
export GEMINI_API_KEY=YOUR_GEMINI_API_KEY
export PYTHONNOUSERSITE=1  # 避免包冲突
```

### 依赖安装
```bash
# 使用安装脚本
./install.sh

# 或手动安装
pip install -r requirements.txt
```

## 输出目录结构

```
outputs/
├── image_generation/     # 图像生成结果
│   ├── gen_img/         # 生成的图像
│   └── edit_img/        # 编辑的图像
├── grounded_sam/         # SAM分割结果
│   └── segmentation/    # 分割输出
├── synthesis/           # 合成结果
├── reflector/           # Reflector输出和中间文件
└── mask/                # World Model掩码输出
    └── run_YYYYMMDD_HHMMSS/  # 时间戳子文件夹
        ├── original.png      # 原始图像
        ├── edited.png        # 编辑图像
        ├── original_mask.png # 原始掩码
        └── edited_mask.png   # 编辑掩码
```

## 模式配置

### 1. 完整模式（推荐）
- ✅ 指令增强
- ✅ 验证和反思
- ✅ 掩码生成
- ✅ 中间文件清理

### 2. 简单模式
- ❌ 指令增强
- ❌ 验证和反思
- ❌ 掩码生成
- ✅ 中间文件清理

### 3. 增强模式
- ✅ 指令增强
- ❌ 验证和反思
- ❌ 掩码生成
- ✅ 中间文件清理

### 4. 反射器模式
- ❌ 指令增强
- ✅ 验证和反思
- ❌ 掩码生成
- ✅ 中间文件清理

## 技术特性

### AI模型
- **文本处理**: `gemini-2.5-pro`
- **图像生成**: 自动选择最佳可用模型
- **图像分析**: `gemini-1.5-flash-latest`

### 验证和反思
- **迭代验证**: 最多3次迭代改进
- **AI反馈**: Gemini提供详细反馈
- **自动修正**: 基于反馈自动调整指令

### 物体检测和分割
- **Grounded SAM 2**: 高精度物体分割
- **目标提取**: 智能提取移动物体
- **掩码生成**: 黑白掩码输出

## 错误处理

- 如果掩码生成失败，会显示警告但继续执行
- 如果目标物体提取失败，会跳过掩码生成
- 如果中间文件清理失败，会显示警告但不影响主流程
- 如果API密钥未设置，会抛出明确的错误信息

## 性能优化

- 自动选择最佳AI模型
- 智能缓存中间结果
- 可配置的迭代次数
- 自动清理临时文件

## 故障排除

### 常见问题

1. **ModuleNotFoundError**: 确保安装了所有依赖
2. **API Key错误**: 检查GEMINI_API_KEY环境变量
3. **CUDA错误**: 确保PyTorch与CUDA版本匹配
4. **内存不足**: 减少max_iterations或关闭掩码生成

### 调试模式

启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 更新日志

- **v1.0**: 基础图像编辑功能
- **v1.1**: 添加Grounded SAM集成
- **v1.2**: 添加迭代验证和反思
- **v1.3**: 添加掩码生成和输出管理
- **v1.4**: 添加多种模式配置选项