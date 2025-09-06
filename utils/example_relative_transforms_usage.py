#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相对透视变换计算脚本使用示例

演示如何使用calculate_relative_transforms.py脚本计算匹配模态对之间的相对透视变换关系
"""

import os
import subprocess
import sys


def run_relative_transforms_example():
    """运行相对透视变换计算示例"""
    
    print("="*70)
    print("相对透视变换关系计算脚本使用示例")
    print("="*70)
    
    # 示例1：处理所有匹配结果
    print("\n📋 示例1: 处理所有匹配结果")
    print("计算所有场景和模态对的相对透视变换关系")
    example1_cmd = [
        "python", "utils/calculate_relative_transforms.py",
        "--rgb_dir", "/mnt/mDisk2/APIDIS_P/mp4",
        "--mm_dir", "/mnt/mDisk2/APIDIS_P/mm", 
        "--output_dir", "/mnt/mDisk2/APIDIS_P/MultiModal_Output"
    ]
    print(f"命令: {' '.join(example1_cmd)}")
    
    # 示例2：处理指定场景
    print("\n📋 示例2: 处理指定场景")
    print("只处理A1和A2场景的匹配结果")
    example2_cmd = [
        "python", "utils/calculate_relative_transforms.py",
        "--rgb_dir", "/mnt/mDisk2/APIDIS_P/mp4",
        "--mm_dir", "/mnt/mDisk2/APIDIS_P/mm",
        "--output_dir", "/mnt/mDisk2/APIDIS_P/MultiModal_Output",
        "--scenes", "A1", "A2"
    ]
    print(f"命令: {' '.join(example2_cmd)}")
    
    # 示例3：处理单个场景
    print("\n📋 示例3: 处理单个场景")
    print("只处理A1场景的所有匹配对")
    example3_cmd = [
        "python", "utils/calculate_relative_transforms.py",
        "--rgb_dir", "/mnt/mDisk2/APIDIS_P/mp4",
        "--mm_dir", "/mnt/mDisk2/APIDIS_P/mm",
        "--output_dir", "/mnt/mDisk2/APIDIS_P/MultiModal_Output",
        "--scenes", "A1"
    ]
    print(f"命令: {' '.join(example3_cmd)}")
    
    # 数学原理说明
    math_explanation = """
🧮 数学原理说明:

假设原始视角为I，两个模态经过透视变换后：
- modality1 = I × H1  (第一个模态经过透视变换H1)
- modality2 = I × H2  (第二个模态经过透视变换H2)

要将modality1对齐到modality2，需要的相对变换关系为：
H_relative = H2 × H1^(-1)

这样：modality1 × H_relative = (I × H1) × (H2 × H1^(-1)) = I × H2 = modality2

相对变换矩阵H_relative就是我们要保存的标签，用于后续的匹配准确度计算。
"""

    file_structure_explanation = """
📁 输出文件结构:

对于匹配对 FreqDomain_vs_MotionThermal，将在以下位置生成文件：
/mnt/mDisk2/APIDIS_P/MultiModal_Output/A1/FreqDomain_vs_MotionThermal/
├── relative_transform_matrix.npy     # 3x3相对透视变换矩阵
└── relative_transform_params.json    # 详细参数和元数据

参数文件包含：
- 场景和模态信息
- 矩阵条件数（数值稳定性指标）
- 变换距离（变换强度度量）
- 矩阵行列式（变换类型指标）
- 计算时间戳
"""

    usage_workflow = """
🔄 完整工作流程:

1. 使用perspective_augmentation.py对多模态视频进行透视变换
2. 使用generate_identity_matrices.py为RGB视频生成单位矩阵
3. 运行原有的视频匹配脚本生成匹配结果
4. 使用calculate_relative_transforms.py计算相对变换关系作为标签
5. 在匹配准确度计算中使用相对变换矩阵作为真值标签

这样可以在有透视变换的情况下正确评估匹配算法的性能。
"""

    print(math_explanation)
    print(file_structure_explanation)
    print(usage_workflow)
    
    print("\n" + "="*70)
    print("📁 文件依赖说明:")
    print("输入文件:")
    print("   - RGB目录下的 *_perspective_matrix.npy 文件")
    print("   - 多模态目录各场景子文件夹下的 *_perspective_matrix.npy 文件")
    print("   - 已存在的匹配结果目录结构")
    print("\n输出文件:")
    print("   - relative_transform_matrix.npy: 相对透视变换矩阵（作为标签使用）")
    print("   - relative_transform_params.json: 详细的计算参数和元数据")
    print("="*70)
    
    # 提供交互式选择
    print("\n🚀 是否要运行某个示例？")
    print("1. 查看帮助信息")
    print("2. 测试脚本（仅显示扫描结果，不实际计算）")
    print("3. 退出")
    
    try:
        choice = input("\n请选择 (1-3): ").strip()
        
        if choice == "1":
            # 显示帮助信息
            help_cmd = ["python", "utils/calculate_relative_transforms.py", "--help"]
            print(f"\n执行: {' '.join(help_cmd)}")
            result = subprocess.run(help_cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("错误:", result.stderr)
                
        elif choice == "2":
            # 测试脚本功能（模拟扫描）
            print("\n🧪 测试脚本功能...")
            
            # 检查目录是否存在
            test_dirs = [
                "/mnt/mDisk2/APIDIS_P/mp4",
                "/mnt/mDisk2/APIDIS_P/mm", 
                "/mnt/mDisk2/APIDIS_P/MultiModal_Output"
            ]
            
            all_exist = True
            for test_dir in test_dirs:
                if os.path.exists(test_dir):
                    print(f"✅ 目录存在: {test_dir}")
                else:
                    print(f"❌ 目录不存在: {test_dir}")
                    all_exist = False
            
            if all_exist:
                print("\n📂 目录结构检查通过，可以运行相对变换计算脚本")
                print("提示：运行示例1命令来处理所有匹配结果")
            else:
                print("\n⚠️ 部分目录不存在，请检查路径配置")
                
        elif choice == "3":
            print("👋 再见！")
        else:
            print("❌ 无效选择")
            
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    run_relative_transforms_example()
