#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量视频模态生成脚本
使用pseudo_multimodal3.py来批量处理文件夹下的视频并生成模态

用法：
python batch_multimodal.py 输入文件夹 输出文件夹 [选项]
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import time

# 导入pseudo_multimodal3.py中的函数
try:
    from pseudo_multimodal3 import process_video_enhanced, ensure_dir
except ImportError:
    print("错误: 无法导入pseudo_multimodal3.py，请确保该文件在同一目录下")
    sys.exit(1)

def get_video_files(folder_path):
    """
    获取文件夹下的所有视频文件
    
    Args:
        folder_path: 视频文件夹路径
        
    Returns:
        list: 视频文件路径列表
    """
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

def process_single_video(video_path, output_dir, variants, resize_width, 
                        temporal_smooth_sigma, motion_buffer_size, flow_buffer_size, 
                        smooth_method):
    """
    处理单个视频
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        variants: 模态类型列表
        resize_width: 调整宽度
        temporal_smooth_sigma: 时间平滑参数
        motion_buffer_size: 运动缓冲区大小
        flow_buffer_size: 光流缓冲区大小
        smooth_method: 平滑方法
        
    Returns:
        bool: 是否成功处理
    """
    try:
        print(f"\n🔍 处理视频: {video_path.name}")
        
        # 创建该视频的输出目录
        video_name = video_path.stem
        video_output_dir = Path(output_dir) / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 调用pseudo_multimodal3.py的处理函数
        process_video_enhanced(
            input_path=str(video_path),
            output_dir=str(video_output_dir),
            resize_width=resize_width,
            variants=variants,
            temporal_smooth_sigma=temporal_smooth_sigma,
            motion_buffer_size=motion_buffer_size,
            flow_buffer_size=flow_buffer_size,
            smooth_method=smooth_method
        )
        
        print(f"✅ 完成: {video_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ 处理失败 {video_path.name}: {e}")
        return False

def batch_process(input_dir, output_dir, variants="all", resize_width=None,
                 temporal_smooth_sigma=1.0, motion_buffer_size=5, flow_buffer_size=3,
                 smooth_method="vectorized"):
    """
    批量处理视频
    
    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
        variants: 模态类型
        resize_width: 调整宽度
        temporal_smooth_sigma: 时间平滑参数
        motion_buffer_size: 运动缓冲区大小
        flow_buffer_size: 光流缓冲区大小
        smooth_method: 平滑方法
    """
    print("🎬 批量视频模态生成工具")
    print("=" * 60)
    print(f"📁 输入目录: {input_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🎯 模态类型: {variants}")
    print(f"📐 调整宽度: {resize_width if resize_width else '保持原始尺寸'}")
    print(f"⏱️  时间平滑: {temporal_smooth_sigma}")
    print(f"🔄 平滑方法: {smooth_method}")
    print("=" * 60)
    
    # 获取所有视频文件
    video_files = get_video_files(input_dir)
    
    if not video_files:
        print("❌ 在指定目录中未找到视频文件")
        return
    
    print(f"📊 找到 {len(video_files)} 个视频文件")
    
    # 确保输出目录存在
    ensure_dir(output_dir)
    
    # 处理模态类型
    if variants.lower() == "all":
        variants_list = ("enhanced_motion_thermal", "temporal_gradient", 
                        "frequency_domain", "texture_removal", "enhanced_flow")
    else:
        variants_list = tuple([v.strip() for v in variants.split(",") if v.strip() != ""])
    
    print(f"🎯 将生成以下模态: {', '.join(variants_list)}")
    print("=" * 60)
    
    # 统计信息
    total_videos = len(video_files)
    success_count = 0
    failed_count = 0
    
    # 批量处理视频
    start_time = time.time()
    
    for i, video_file in enumerate(tqdm(video_files, desc="处理进度")):
        print(f"\n[{i+1}/{total_videos}] 处理: {video_file.name}")
        
        success = process_single_video(
            video_path=video_file,
            output_dir=output_dir,
            variants=variants_list,
            resize_width=resize_width,
            temporal_smooth_sigma=temporal_smooth_sigma,
            motion_buffer_size=motion_buffer_size,
            flow_buffer_size=flow_buffer_size,
            smooth_method=smooth_method
        )
        
        if success:
            success_count += 1
        else:
            failed_count += 1
    
    # 显示统计结果
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("📊 批量处理完成统计")
    print("=" * 60)
    print(f"📹 总视频数: {total_videos}")
    print(f"✅ 成功处理: {success_count}")
    print(f"❌ 处理失败: {failed_count}")
    print(f"⏱️  总耗时: {total_time:.2f} 秒")
    print(f"📊 平均每视频: {total_time/total_videos:.2f} 秒")
    print("=" * 60)
    
    if success_count > 0:
        print(f"🎉 成功生成模态的视频已保存到: {output_dir}")
        print("📁 每个视频都有独立的子文件夹，包含所有模态类型")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量视频模态生成工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python batch_multimodal.py ./videos ./output
  python batch_multimodal.py ./videos ./output --variants all --resize 640
  python batch_multimodal.py ./videos ./output --variants enhanced_motion_thermal,temporal_gradient
        """
    )
    
    parser.add_argument("input_dir", help="输入视频文件夹路径")
    parser.add_argument("output_dir", help="输出文件夹路径")
    parser.add_argument("--variants", default="all", 
                       help="模态类型: all 或 逗号分隔的列表 (默认: all)")
    parser.add_argument("--resize", type=int, default=None,
                       help="调整视频宽度，保持宽高比 (可选)")
    parser.add_argument("--temporal_smooth", type=float, default=1.0,
                       help="时间平滑参数 (默认: 1.0)")
    parser.add_argument("--motion_buffer", type=int, default=5,
                       help="运动缓冲区大小 (默认: 5)")
    parser.add_argument("--flow_buffer", type=int, default=3,
                       help="光流缓冲区大小 (默认: 3)")
    parser.add_argument("--smooth_method", default="vectorized",
                       choices=["vectorized", "parallel", "separable"],
                       help="平滑方法 (默认: vectorized)")
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        sys.exit(1)
    
    try:
        batch_process(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            variants=args.variants,
            resize_width=args.resize,
            temporal_smooth_sigma=args.temporal_smooth,
            motion_buffer_size=args.motion_buffer,
            flow_buffer_size=args.flow_buffer,
            smooth_method=args.smooth_method
        )
    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
