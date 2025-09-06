#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示透视变换脚本如何保持目录结构和文件命名的一致性
"""

import os
from pathlib import Path

def create_demo_structure():
    """创建演示的输入/输出目录结构对比"""
    
    print("🎯 透视变换数据增强 - 目录结构保持演示")
    print("="*70)
    
    # 模拟输入目录结构
    input_structure = """
📁 输入目录结构 (/mnt/mDisk2/APIDIS/mm):
/mnt/mDisk2/APIDIS/mm/
├── A1/
│   ├── enhanced_motion_thermal.mp4
│   ├── temporal_gradient.mp4
│   ├── motion_saliency.mp4
│   ├── frequency_domain.mp4
│   └── texture_removal.mp4
├── A2/
│   ├── enhanced_motion_thermal.mp4
│   ├── temporal_gradient.mp4
│   └── enhanced_flow.mp4
└── A3/
    ├── motion_saliency.mp4
    └── frequency_domain.mp4
"""

    output_structure = """
📁 输出目录结构 (指定的输出目录):
your_output_directory/
├── A1/
│   ├── enhanced_motion_thermal.mp4              # ✅ 保持原始文件名
│   ├── enhanced_motion_thermal_perspective_matrix.npy
│   ├── enhanced_motion_thermal_perspective_params.json
│   ├── temporal_gradient.mp4                    # ✅ 保持原始文件名
│   ├── temporal_gradient_perspective_matrix.npy
│   ├── temporal_gradient_perspective_params.json
│   ├── motion_saliency.mp4                      # ✅ 保持原始文件名
│   ├── motion_saliency_perspective_matrix.npy
│   ├── motion_saliency_perspective_params.json
│   ├── frequency_domain.mp4                     # ✅ 保持原始文件名
│   ├── frequency_domain_perspective_matrix.npy
│   ├── frequency_domain_perspective_params.json
│   ├── texture_removal.mp4                      # ✅ 保持原始文件名
│   ├── texture_removal_perspective_matrix.npy
│   └── texture_removal_perspective_params.json
├── A2/
│   ├── enhanced_motion_thermal.mp4              # ✅ 保持原始文件名
│   ├── enhanced_motion_thermal_perspective_matrix.npy
│   ├── enhanced_motion_thermal_perspective_params.json
│   ├── temporal_gradient.mp4                    # ✅ 保持原始文件名
│   ├── temporal_gradient_perspective_matrix.npy
│   ├── temporal_gradient_perspective_params.json
│   ├── enhanced_flow.mp4                        # ✅ 保持原始文件名
│   ├── enhanced_flow_perspective_matrix.npy
│   └── enhanced_flow_perspective_params.json
└── A3/
    ├── motion_saliency.mp4                      # ✅ 保持原始文件名
    ├── motion_saliency_perspective_matrix.npy
    ├── motion_saliency_perspective_params.json
    ├── frequency_domain.mp4                     # ✅ 保持原始文件名
    ├── frequency_domain_perspective_matrix.npy
    └── frequency_domain_perspective_params.json
"""

    usage_examples = """
🚀 使用示例:

1. 基本使用（保持结构和命名一致）:
   python utils/perspective_augmentation.py \\
       --input_dir /mnt/mDisk2/APIDIS/mm \\
       --output_dir /mnt/mDisk2/APIDIS/mm_perspective_augmented

2. 处理单个场景（保持结构和命名一致）:
   python utils/perspective_augmentation.py \\
       --input_dir /mnt/mDisk2/APIDIS/mm/A1 \\
       --output_dir /path/to/output/A1

3. 自定义变换强度:
   python utils/perspective_augmentation.py \\
       --input_dir /mnt/mDisk2/APIDIS/mm \\
       --output_dir /mnt/mDisk2/APIDIS/mm_augmented \\
       --max_shift 0.03 \\
       --seed 42
"""

    key_features = """
✨ 关键特性:

✅ 完全保持目录结构  : A1/file.mp4 → output/A1/file.mp4
✅ 保持原始文件名    : enhanced_motion_thermal.mp4 → enhanced_motion_thermal.mp4
✅ 自动创建子目录    : 输出目录中自动重建所有子目录
✅ 透视变换矩阵保存  : 每个视频对应一个.npy矩阵文件
✅ 详细参数记录      : 每个视频对应一个.json参数文件
✅ 批量处理能力      : 一次处理整个目录树
✅ 进度显示          : 实时显示处理进度和映射关系

⚠️ 注意事项:
- 输出目录中的视频文件将完全替换为透视变换后的版本
- 透视变换矩阵和参数文件作为额外文件添加到输出目录
- 建议使用不同的输出目录避免覆盖原始数据
"""

    print(input_structure)
    print(output_structure)
    print(usage_examples)
    print(key_features)
    print("="*70)

if __name__ == "__main__":
    create_demo_structure()
