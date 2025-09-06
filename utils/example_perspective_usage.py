#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
透视变换脚本使用示例

演示如何使用perspective_augmentation.py脚本对多模态视频进行数据增强
"""

import os
import subprocess
import sys


def run_perspective_augmentation_example():
    """运行透视变换数据增强示例"""
    
    print("="*60)
    print("透视变换数据增强脚本使用示例")
    print("="*60)
    
    # 示例1：基础用法
    print("\n📋 示例1: 基础用法")
    print("对 /mnt/mDisk2/APIDIS/mm 目录下的所有视频进行透视变换")
    example1_cmd = [
        "python", "utils/perspective_augmentation.py",
        "--input_dir", "/mnt/mDisk2/APIDIS/mm"
    ]
    print(f"命令: {' '.join(example1_cmd)}")
    
    # 示例2：指定输出目录
    print("\n📋 示例2: 指定输出目录")
    print("将增强后的视频保存到指定目录")
    example2_cmd = [
        "python", "utils/perspective_augmentation.py", 
        "--input_dir", "/mnt/mDisk2/APIDIS/mm",
        "--output_dir", "/mnt/mDisk2/APIDIS/mm_perspective_augmented"
    ]
    print(f"命令: {' '.join(example2_cmd)}")
    
    # 示例3：调整变换强度
    print("\n📋 示例3: 调整变换强度")
    print("使用较小的变换强度（3%偏移）")
    example3_cmd = [
        "python", "utils/perspective_augmentation.py",
        "--input_dir", "/mnt/mDisk2/APIDIS/mm",
        "--max_shift", "0.03"
    ]
    print(f"命令: {' '.join(example3_cmd)}")
    
    # 示例4：可重现的结果
    print("\n📋 示例4: 设置随机种子获得可重现结果")
    print("使用固定随机种子")
    example4_cmd = [
        "python", "utils/perspective_augmentation.py",
        "--input_dir", "/mnt/mDisk2/APIDIS/mm",
        "--seed", "42"
    ]
    print(f"命令: {' '.join(example4_cmd)}")
    
    # 示例5：处理单个子文件夹
    print("\n📋 示例5: 处理单个场景")
    print("只处理A1场景的多模态数据")
    example5_cmd = [
        "python", "utils/perspective_augmentation.py",
        "--input_dir", "/mnt/mDisk2/APIDIS/mm/A1",
        "--output_dir", "/mnt/mDisk2/APIDIS/mm_augmented/A1"
    ]
    print(f"命令: {' '.join(example5_cmd)}")
    
    print("\n" + "="*60)
    print("📁 输出文件说明:")
    print("   对于输入视频 'video.mp4'，将生成：")
    print("   - video.mp4                      : 透视变换后的视频（保持原始文件名）")
    print("   - video_perspective_matrix.npy   : 透视变换矩阵（3x3）")
    print("   - video_perspective_params.json  : 变换参数记录")
    print("   ✅ 完全保持输入目录的结构和文件命名")
    print("="*60)
    
    # 提供交互式选择
    print("\n🚀 是否要运行某个示例？")
    print("1. 查看帮助信息")
    print("2. 运行示例3（小变换强度）在测试目录")
    print("3. 退出")
    
    try:
        choice = input("\n请选择 (1-3): ").strip()
        
        if choice == "1":
            # 显示帮助信息
            help_cmd = ["python", "utils/perspective_augmentation.py", "--help"]
            print(f"\n执行: {' '.join(help_cmd)}")
            result = subprocess.run(help_cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("错误:", result.stderr)
                
        elif choice == "2":
            # 创建测试目录和视频
            test_dir = "/tmp/test_perspective_augmentation"
            os.makedirs(test_dir, exist_ok=True)
            
            # 创建一个简单的测试视频（如果OpenCV可用）
            try:
                import cv2
                import numpy as np
                
                # 创建测试视频
                test_video_path = os.path.join(test_dir, "test_video.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(test_video_path, fourcc, 25.0, (640, 480))
                
                # 生成30帧测试视频
                for i in range(30):
                    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                    # 添加一些几何图形便于观察变换效果
                    cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), 3)
                    cv2.circle(frame, (400, 300), 50, (0, 255, 0), 3)
                    cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    out.write(frame)
                
                out.release()
                
                print(f"\n✅ 创建测试视频: {test_video_path}")
                
                # 运行透视变换
                test_cmd = [
                    "python", "utils/perspective_augmentation.py",
                    "--input_dir", test_dir,
                    "--max_shift", "0.08",
                    "--seed", "123"
                ]
                
                print(f"执行: {' '.join(test_cmd)}")
                result = subprocess.run(test_cmd)
                
                if result.returncode == 0:
                    print(f"\n✅ 测试完成！检查输出目录: {test_dir}/perspective_augmented")
                else:
                    print(f"\n❌ 测试失败，返回码: {result.returncode}")
                    
            except ImportError:
                print("❌ 无法导入OpenCV，跳过测试视频创建")
            except Exception as e:
                print(f"❌ 创建测试视频时出错: {e}")
                
        elif choice == "3":
            print("👋 再见！")
        else:
            print("❌ 无效选择")
            
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")


if __name__ == "__main__":
    run_perspective_augmentation_example()
