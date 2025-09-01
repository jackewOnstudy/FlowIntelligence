#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频批量重命名脚本
将文件夹下的所有视频按顺序重命名为V1、V2、V3...

用法：
python rename_videos.py 视频文件夹路径
"""

import os
import sys
from pathlib import Path

def rename_videos(folder_path):
    """
    重命名文件夹下的所有视频文件
    
    Args:
        folder_path: 视频文件夹路径
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"错误: 文件夹不存在: {folder_path}")
        return
    
    if not folder.is_dir():
        print(f"错误: 不是文件夹: {folder_path}")
        return
    
    # 支持的视频格式
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.MP4', '.AVI', '.MOV', '.MKV', '.WMV', '.FLV', '.WEBM'}
    
    # 获取所有视频文件
    video_files = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix in video_extensions:
            video_files.append(file)
    
    if not video_files:
        print(f"在文件夹 {folder_path} 中未找到视频文件")
        return
    
    # 按文件名排序
    video_files.sort(key=lambda x: x.name)
    
    print(f"找到 {len(video_files)} 个视频文件:")
    for i, file in enumerate(video_files, 1):
        print(f"  {i}. {file.name}")
    
    print(f"\n开始重命名...")
    
    # 重命名文件
    for i, old_file in enumerate(video_files, 1):
        # 保持原扩展名
        extension = old_file.suffix
        new_name = f"A{i}{extension}"
        new_path = old_file.parent / new_name
        
        try:
            # 如果新文件名已存在，先重命名为临时名称
            if new_path.exists():
                temp_name = f"temp_{i}{extension}"
                temp_path = old_file.parent / temp_name
                old_file.rename(temp_path)
                old_file = temp_path
            
            # 重命名为最终名称
            old_file.rename(new_path)
            print(f"  ✅ {old_file.name} -> {new_name}")
            
        except Exception as e:
            print(f"  ❌ 重命名 {old_file.name} 失败: {e}")
    
    print(f"\n✅ 重命名完成! 共处理 {len(video_files)} 个视频文件")

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("用法: python rename_videos.py 视频文件夹路径")
        print("示例: python rename_videos.py ./videos")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    print("🎬 视频批量重命名工具")
    print(f"目标文件夹: {folder_path}")
    print("="*50)
    
    try:
        rename_videos(folder_path)
    except KeyboardInterrupt:
        print("\n\n⏹️ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
