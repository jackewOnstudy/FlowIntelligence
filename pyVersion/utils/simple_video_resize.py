#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版视频批量resize脚本
专门用于以下resize需求：
- 3840x2160 -> 1920x1080
- 2160x3840 -> 1080x1920

用法：
python simple_video_resize.py 输入文件夹路径 输出文件夹路径
"""

import os
import cv2
import sys
from pathlib import Path

def resize_video(input_path, output_path, target_width, target_height):
    """
    对视频进行resize
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        target_width: 目标宽度
        target_height: 目标高度
    """
    print(f"处理: {os.path.basename(input_path)}")
    
    # 打开输入视频
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"  无法打开视频: {input_path}")
        return False
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (target_width, target_height))
    
    if not writer.isOpened():
        print(f"  无法创建输出视频: {output_path}")
        cap.release()
        return False
    
    # 处理每一帧
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # resize帧
        resized_frame = cv2.resize(frame, (target_width, target_height))
        
        # 写入帧
        writer.write(resized_frame)
        frame_count += 1
        
        # 显示进度
        if frame_count % 100 == 0:
            print(f"    已处理 {frame_count} 帧")
    
    # 释放资源
    cap.release()
    writer.release()
    
    print(f"  完成! 处理了 {frame_count} 帧")
    return True

def process_folder(input_dir, output_dir):
    """
    处理文件夹下的所有视频
    
    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 支持的视频格式
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    total_videos = 0
    processed_4k = 0
    processed_9_16 = 0
    unchanged = 0
    
    # 遍历所有视频文件
    for video_file in input_path.rglob("*"):
        if video_file.suffix.lower() in video_extensions:
            total_videos += 1
            
            # 获取视频信息
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                print(f"跳过无法读取的视频: {video_file.name}")
                continue
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            print(f"\n视频: {video_file.name}")
            print(f"  原始分辨率: {width}x{height}")
            
            # 判断是否需要resize
            if width == 3840 and height == 2160:
                # 4K横屏 -> 1080p横屏
                target_width, target_height = 1920, 1080
                processed_4k += 1
                print(f"  需要resize到: {target_width}x{target_height}")
                
            elif width == 2160 and (height == 3840 or height == 3804):
                # 9:16竖屏 -> 1080x1920
                target_width, target_height = 1080, 1920
                processed_9_16 += 1
                print(f"  需要resize到: {target_width}x{target_height}")

            elif width == 2560 and height == 1440:
                # 2K横屏 -> 1080p横屏
                target_width, target_height = 1920, 1080
                processed_4k += 1
                print(f"  需要resize到: {target_width}x{target_height}")

            elif width == 1440 and height == 2560:
                # 9:16竖屏 2K -> 1080x1920
                target_width, target_height = 1080, 1920
                processed_9_16 += 1
                print(f"  需要resize到: {target_width}x{target_height}")
                
            else:
                # 其他分辨率保持不变
                print(f"  无需resize，保持原始分辨率")
                unchanged += 1
                continue
            
            # 构建输出路径
            relative_path = video_file.relative_to(input_path)
            output_file = output_path / relative_path
            
            # 执行resize
            if resize_video(video_file, output_file, target_width, target_height):
                print(f"  保存到: {output_file}")
            else:
                print(f"  处理失败: {video_file.name}")
    
    # 显示统计结果
    print(f"\n{'='*50}")
    print("处理完成统计:")
    print(f"总视频数: {total_videos}")
    print(f"4K横屏resize (3840x2160->1920x1080): {processed_4k}")
    print(f"9:16竖屏resize (2160x3840->1080x1920): {processed_9_16}")
    print(f"保持不变: {unchanged}")
    print(f"{'='*50}")

def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("用法: python simple_video_resize.py 输入文件夹 输出文件夹")
        print("示例: python simple_video_resize.py ./videos ./resized_videos")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not os.path.exists(input_dir):
        print(f"错误: 输入文件夹不存在: {input_dir}")
        sys.exit(1)
    
    print("🎬 视频批量Resize工具")
    print(f"输入文件夹: {input_dir}")
    print(f"输出文件夹: {output_dir}")
    print("="*50)
    
    try:
        process_folder(input_dir, output_dir)
        print("\n✅ 所有视频处理完成!")
    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
