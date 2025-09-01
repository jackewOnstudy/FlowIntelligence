#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版批量视频模态生成脚本
使用pseudo_multimodal3.py来批量处理文件夹下的视频并生成模态

用法：
python simple_batch_multimodal.py 输入文件夹 输出文件夹
"""

import os
import sys
from pathlib import Path

# 导入pseudo_multimodal3.py中的函数
try:
    from pseudo_multimodal3 import process_video_enhanced, ensure_dir
except ImportError:
    print("错误: 无法导入pseudo_multimodal3.py，请确保该文件在同一目录下")
    sys.exit(1)

def get_video_files(folder_path):
    """获取文件夹下的所有视频文件（不包含子目录）"""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"错误: 文件夹不存在: {folder_path}")
        return []
    
    if not folder.is_dir():
        print(f"错误: 不是文件夹: {folder_path}")
        return []
    
    # 支持的视频格式
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', 
                        '.MP4', '.AVI', '.MOV', '.MKV', '.WMV', '.FLV', '.WEBM'}
    
    # 获取所有视频文件（不包含子目录）
    video_files = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix in video_extensions:
            video_files.append(file)
    
    return sorted(video_files)

def process_video(video_path, output_dir):
    """处理单个视频"""
    try:
        print(f"处理: {video_path.name}")
        
        # 创建该视频的输出目录
        video_name = video_path.stem
        video_output_dir = Path(output_dir) / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 调用pseudo_multimodal3.py的处理函数，生成所有模态
        process_video_enhanced(
            input_path=str(video_path),
            output_dir=str(video_output_dir),
            resize_width=None,  # 保持原始尺寸
            variants=("enhanced_motion_thermal", "temporal_gradient", 
                     "frequency_domain", "texture_removal", "enhanced_flow"),
            temporal_smooth_sigma=1.0,
            motion_buffer_size=5,
            flow_buffer_size=3,
            smooth_method="vectorized"
        )
        
        print(f"✅ 完成: {video_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ 处理失败 {video_path.name}: {e}")
        return False

def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("用法: python simple_batch_multimodal.py 输入文件夹 输出文件夹")
        print("示例: python simple_batch_multimodal.py ./videos ./output")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(input_dir):
        print(f"错误: 输入文件夹不存在: {input_dir}")
        sys.exit(1)
    
    print("🎬 批量视频模态生成工具")
    print(f"输入文件夹: {input_dir}")
    print(f"输出文件夹: {output_dir}")
    print("="*50)
    
    # 获取所有视频文件
    video_files = get_video_files(input_dir)
    
    if not video_files:
        print("❌ 在指定目录中未找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件:")
    for i, file in enumerate(video_files, 1):
        print(f"  {i}. {file.name}")
    
    print(f"\n开始处理...")
    print("="*50)
    
    # 确保输出目录存在
    ensure_dir(output_dir)
    
    # 批量处理视频
    success_count = 0
    failed_count = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] 处理: {video_file.name}")
        
        if process_video(video_file, output_dir):
            success_count += 1
        else:
            failed_count += 1
    
    # 显示结果
    print(f"\n{'='*50}")
    print("处理完成统计:")
    print(f"总视频数: {len(video_files)}")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"{'='*50}")
    
    if success_count > 0:
        print(f"🎉 成功生成模态的视频已保存到: {output_dir}")
        print("📁 每个视频都有独立的子文件夹，包含以下模态:")
        print("  - enhanced_motion_thermal (增强运动热图)")
        print("  - temporal_gradient (时间梯度)")
        print("  - frequency_domain (频域变换)")
        print("  - texture_removal (纹理移除)")
        print("  - enhanced_flow (增强光流)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        sys.exit(1)
