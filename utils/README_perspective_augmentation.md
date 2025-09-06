# 多模态视频透视变换数据增强工具

## 📖 简介

这个工具专为多模态视频特征匹配任务设计，可以对视频进行适度的透视变换数据增强，同时保存变换矩阵作为后续标签使用。

## 🎯 主要功能

- ✅ **递归处理**: 自动遍历文件夹及子文件夹中的所有视频文件
- ✅ **适度变换**: 生成合理的透视变换，避免过强的缩放关系
- ✅ **保持结构**: 保持原有的文件夹结构
- ✅ **矩阵保存**: 保存透视变换矩阵和详细参数用作标签
- ✅ **进度显示**: 实时显示处理进度和统计信息
- ✅ **可重现**: 支持随机种子设置，确保结果可重现

## 📂 文件说明

```
utils/
├── perspective_augmentation.py      # 主要脚本
├── example_perspective_usage.py     # 使用示例
└── README_perspective_augmentation.md # 说明文档（本文件）
```

## 🚀 快速开始

### 基础用法

```bash
# 对指定目录进行透视变换增强
python utils/perspective_augmentation.py --input_dir /mnt/mDisk2/APIDIS/mm
```

### 完整参数

```bash
python utils/perspective_augmentation.py \
    --input_dir /mnt/mDisk2/APIDIS/mm \
    --output_dir /mnt/mDisk2/APIDIS/mm_augmented \
    --max_shift 0.05 \
    --seed 42
```

## 📋 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_dir` | ✅ | - | 输入目录路径（包含视频文件） |
| `--output_dir` | ❌ | `{input_dir}/perspective_augmented` | 输出目录路径 |
| `--max_shift` | ❌ | `0.05` | 最大偏移比例（相对于图像尺寸） |
| `--seed` | ❌ | 随机 | 随机种子，用于可重现结果 |

### max_shift 参数说明

- `0.01` - 非常轻微的变换（1%偏移）
- `0.05` - 适度变换（5%偏移，推荐）
- `0.1` - 较强变换（10%偏移）
- `0.2` - 很强变换（20%偏移，不推荐）

## 📁 输出文件结构

**重要**: 输出目录将完全保持与输入目录相同的结构和文件命名！

对于输入目录结构：
```
/mnt/mDisk2/APIDIS/mm/
├── A1/
│   ├── enhanced_motion_thermal.mp4
│   ├── temporal_gradient.mp4
│   └── frequency_domain.mp4
└── A2/
    ├── enhanced_motion_thermal.mp4
    └── texture_removal.mp4
```

输出目录结构将为：
```
output_dir/
├── A1/
│   ├── enhanced_motion_thermal.mp4         # 透视变换后的视频（保持原始文件名）
│   ├── enhanced_motion_thermal_perspective_matrix.npy
│   ├── enhanced_motion_thermal_perspective_params.json
│   ├── temporal_gradient.mp4               # 透视变换后的视频（保持原始文件名）
│   ├── temporal_gradient_perspective_matrix.npy
│   ├── temporal_gradient_perspective_params.json
│   ├── frequency_domain.mp4                # 透视变换后的视频（保持原始文件名）
│   ├── frequency_domain_perspective_matrix.npy
│   └── frequency_domain_perspective_params.json
└── A2/
    ├── enhanced_motion_thermal.mp4         # 透视变换后的视频（保持原始文件名）
    ├── enhanced_motion_thermal_perspective_matrix.npy
    ├── enhanced_motion_thermal_perspective_params.json
    ├── texture_removal.mp4                 # 透视变换后的视频（保持原始文件名）
    ├── texture_removal_perspective_matrix.npy
    └── texture_removal_perspective_params.json
```

### 输出文件详细说明

1. **原始视频文件名（如`enhanced_motion_thermal.mp4`）**: 应用透视变换后的视频文件，保持原始文件名不变
2. **`*_perspective_matrix.npy`**: 3x3透视变换矩阵，可用于：
   - 将原始坐标转换到变换后坐标
   - 作为深度学习模型的标签
   - 逆变换恢复原始坐标
3. **`*_perspective_params.json`**: 包含详细的变换参数：
   ```json
   {
     "src_points": [[0, 0], [width, 0], [width, height], [0, height]],
     "dst_points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
     "image_size": [width, height],
     "max_shift_ratio": 0.05,
     "max_shift_pixels": [max_w, max_h]
   }
   ```

## 🔧 使用示例

### 示例1: 处理整个多模态数据集

```bash
# 处理 /mnt/mDisk2/APIDIS/mm 下的所有视频
python utils/perspective_augmentation.py \
    --input_dir /mnt/mDisk2/APIDIS/mm \
    --max_shift 0.05 \
    --seed 42
```

### 示例2: 处理单个场景

```bash
# 只处理A1场景的数据
python utils/perspective_augmentation.py \
    --input_dir /mnt/mDisk2/APIDIS/mm/A1 \
    --output_dir /mnt/mDisk2/APIDIS/mm_augmented/A1
```

### 示例3: 批量处理多个场景

```bash
# 创建批处理脚本
for scene in A1 A2 A3; do
    python utils/perspective_augmentation.py \
        --input_dir /mnt/mDisk2/APIDIS/mm/$scene \
        --output_dir /mnt/mDisk2/APIDIS/mm_augmented/$scene \
        --max_shift 0.05 \
        --seed 42
done
```

## 🧮 透视变换矩阵使用

### 在Python中加载和使用矩阵

```python
import numpy as np
import cv2

# 加载透视变换矩阵
matrix = np.load('video_perspective_matrix.npy')

# 应用变换到坐标点
src_points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
dst_points = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), matrix)

# 应用变换到图像
transformed_image = cv2.warpPerspective(image, matrix, (width, height))

# 计算逆变换矩阵
inv_matrix = np.linalg.inv(matrix)
```

### 在特征匹配中使用

```python
# 假设有原始图像的特征点坐标
original_keypoints = np.array([[x1, y1], [x2, y2], ...])

# 转换为齐次坐标
homogeneous_points = np.hstack([original_keypoints, np.ones((len(original_keypoints), 1))])

# 应用透视变换
transformed_points = (matrix @ homogeneous_points.T).T
transformed_keypoints = transformed_points[:, :2] / transformed_points[:, 2:]

# 现在可以用transformed_keypoints作为增强后视频的标签
```

## 📊 性能参考

- **处理速度**: 约 15-30 FPS（取决于视频分辨率和硬件）
- **内存占用**: 约 200-500MB（取决于视频分辨率）
- **支持格式**: MP4, AVI, MOV, MKV, WMV, FLV

## ⚠️ 注意事项

1. **变换强度**: 建议 `max_shift` 保持在 0.03-0.08 之间，避免过强变换
2. **存储空间**: 增强后的视频文件大小与原文件相当，请确保足够存储空间
3. **处理时间**: 大型视频文件处理时间较长，建议在后台运行
4. **视频质量**: 透视变换可能引入轻微的插值误差，属于正常现象

## 🛠️ 故障排除

### 常见问题

1. **"无法打开视频"**
   - 检查视频文件是否损坏
   - 确认OpenCV支持该视频格式
   - 检查文件路径是否正确

2. **"无法创建输出视频"**
   - 检查输出目录写权限
   - 确认磁盘空间充足
   - 检查文件名是否包含特殊字符

3. **处理速度很慢**
   - 减少视频分辨率
   - 使用SSD存储
   - 增加系统内存

### 日志输出说明

- `🎬 处理视频`: 开始处理某个视频文件
- `✅ 视频处理完成`: 成功完成一个视频的处理
- `💾 保存变换数据`: 保存透视变换矩阵和参数
- `📊 处理统计`: 最终的处理统计信息

## 📚 相关文档

- [OpenCV 透视变换文档](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)
- [NumPy 数组操作文档](https://numpy.org/doc/stable/reference/routines.array-manipulation.html)

## 🤝 支持

如有问题或建议，请：
1. 检查本文档的故障排除部分
2. 运行 `python utils/example_perspective_usage.py` 查看示例
3. 使用 `--help` 参数查看完整帮助信息
