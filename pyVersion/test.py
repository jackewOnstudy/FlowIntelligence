#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频帧数检测脚本
使用OpenCV读取视频文件并显示详细的帧数信息
"""

import cv2
import os
import sys
import argparse
from pathlib import Path

def get_video_info(video_path):
    """
    获取视频的详细信息
    
    Args:
        video_path (str): 视频文件路径
        
    Returns:
        dict: 包含视频信息的字典
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    try:
        # 获取视频属性
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 计算视频时长
        duration_seconds = total_frames / fps if fps > 0 else 0
        duration_minutes = duration_seconds / 60
        
        # 获取视频编码格式
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        # 获取文件大小
        file_size = os.path.getsize(video_path)
        file_size_mb = file_size / (1024 * 1024)
        
        info = {
            'path': video_path,
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height,
            'duration_seconds': duration_seconds,
            'duration_minutes': duration_minutes,
            'codec': codec,
            'file_size_mb': file_size_mb
        }
        
        return info
        
    finally:
        cap.release()

def print_video_info(info):
    """
    打印视频信息
    
    Args:
        info (dict): 视频信息字典
    """
    print("=" * 60)
    print("🎬 视频信息分析结果")
    print("=" * 60)
    print(f"📁 文件路径: {info['path']}")
    print(f"📊 文件大小: {info['file_size_mb']:.2f} MB")
    print(f"🎯 总帧数: {info['total_frames']:,} 帧")
    print(f"⚡ 帧率: {info['fps']:.2f} FPS")
    print(f"📐 分辨率: {info['width']} x {info['height']}")
    print(f"⏱️  时长: {info['duration_minutes']:.2f} 分钟 ({info['duration_seconds']:.2f} 秒)")
    print(f"🎞️  编码格式: {info['codec']}")
    print("=" * 60)

def verify_frame_count(video_path, max_frames_to_check=100):
    """
    验证视频帧数（通过实际读取帧来验证）
    
    Args:
        video_path (str): 视频文件路径
        max_frames_to_check (int): 最大检查帧数（避免处理过长的视频）
        
    Returns:
        int: 实际读取到的帧数
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    try:
        actual_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            actual_frames += 1
            
            # 限制检查的帧数，避免处理过长的视频
            if actual_frames >= max_frames_to_check:
                break
                
        return actual_frames
        
    finally:
        cap.release()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="视频帧数检测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python test.py video.mp4                    # 分析单个视频
  python test.py video1.mp4 video2.mp4       # 分析多个视频
  python test.py --verify video.mp4          # 验证帧数
  python test.py --max-check 50 video.mp4    # 限制验证帧数
        """
    )
    
    parser.add_argument('videos', nargs='+', help='要分析的视频文件路径')
    parser.add_argument('--verify', action='store_true', help='验证实际帧数（通过读取帧）')
    parser.add_argument('--max-check', type=int, default=100, 
                       help='验证时的最大检查帧数 (默认: 100)')
    parser.add_argument('--output', '-o', help='输出结果到文件')
    
    args = parser.parse_args()
    
    # 检查视频文件是否存在
    valid_videos = []
    for video_path in args.videos:
        if os.path.exists(video_path):
            valid_videos.append(video_path)
        else:
            print(f"⚠️  警告: 文件不存在: {video_path}")
    
    if not valid_videos:
        print("❌ 没有找到有效的视频文件")
        return 1
    
    results = []
    
    # 分析每个视频
    for video_path in valid_videos:
        try:
            print(f"\n🔍 正在分析视频: {video_path}")
            
            # 获取视频信息
            info = get_video_info(video_path)
            
            # 如果需要验证帧数
            if args.verify:
                print("🔍 正在验证实际帧数...")
                actual_frames = verify_frame_count(video_path, args.max_check)
                info['verified_frames'] = actual_frames
                
                if actual_frames < args.max_check:
                    print(f"✅ 验证完成: 实际读取到 {actual_frames} 帧")
                else:
                    print(f"⚠️  验证完成: 读取到 {actual_frames} 帧 (达到最大检查限制)")
            
            # 打印信息
            print_video_info(info)
            results.append(info)
            
        except Exception as e:
            print(f"❌ 分析视频 {video_path} 时出错: {e}")
            continue
    
    # 输出到文件
    if args.output and results:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write("视频分析结果\n")
                f.write("=" * 60 + "\n\n")
                
                for info in results:
                    f.write(f"文件: {info['path']}\n")
                    f.write(f"总帧数: {info['total_frames']:,} 帧\n")
                    f.write(f"帧率: {info['fps']:.2f} FPS\n")
                    f.write(f"分辨率: {info['width']} x {info['height']}\n")
                    f.write(f"时长: {info['duration_minutes']:.2f} 分钟\n")
                    if 'verified_frames' in info:
                        f.write(f"验证帧数: {info['verified_frames']} 帧\n")
                    f.write("\n")
            
            print(f"\n💾 结果已保存到: {args.output}")
            
        except Exception as e:
            print(f"❌ 保存结果到文件时出错: {e}")
    
    # 显示总结
    if len(results) > 1:
        print("\n📊 总结:")
        print("-" * 40)
        total_frames = sum(info['total_frames'] for info in results)
        total_duration = sum(info['duration_seconds'] for info in results)
        print(f"总视频数: {len(results)}")
        print(f"总帧数: {total_frames:,} 帧")
        print(f"总时长: {total_duration/60:.2f} 分钟")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        sys.exit(1)
