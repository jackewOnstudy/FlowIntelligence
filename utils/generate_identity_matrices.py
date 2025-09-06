#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB视频单位矩阵生成脚本

功能特性:
- 为RGB视频目录下的所有视频生成对应的单位矩阵npy文件
- 保持与透视变换脚本相同的文件命名规范
- 支持递归处理子目录
- 生成与透视变换脚本兼容的参数文件

使用场景:
RGB视频不需要透视变换，但为了统一处理流程，需要生成对应的单位矩阵文件，
这样在后续的特征匹配和数据处理中可以使用相同的代码逻辑。

使用示例:
1. 为RGB视频目录生成单位矩阵:
   python generate_identity_matrices.py --input_dir /mnt/mDisk2/APIDIS/mp4

2. 指定输出目录:
   python generate_identity_matrices.py --input_dir /mnt/mDisk2/APIDIS/mp4 --output_dir /path/to/output

3. 查看帮助信息:
   python generate_identity_matrices.py --help

输出:
- 单位矩阵文件（.npy格式，3x3单位矩阵）
- 参数记录文件（.json格式，记录矩阵信息）
- 保持与输入目录相同的结构和文件命名
"""

import os
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Dict
import glob


def create_identity_matrix(image_width: int, image_height: int) -> tuple:
    """
    创建3x3单位矩阵和对应的参数信息
    
    Args:
        image_width: 图像宽度
        image_height: 图像高度
        
    Returns:
        identity_matrix: 3x3单位矩阵
        params: 参数字典
    """
    # 创建3x3单位矩阵
    identity_matrix = np.eye(3, dtype=np.float32)
    
    # 生成参数信息，保持与透视变换脚本的格式一致
    src_points = np.float32([
        [0, 0],                        # 左上角
        [image_width, 0],              # 右上角  
        [image_width, image_height],   # 右下角
        [0, image_height]              # 左下角
    ])
    
    # 对于单位矩阵，源点和目标点相同
    dst_points = src_points.copy()
    
    params = {
        "src_points": src_points.tolist(),
        "dst_points": dst_points.tolist(),
        "image_size": [image_width, image_height],
        "transform_type": "identity",
        "max_shift_ratio": 0.0,
        "max_shift_pixels": [0, 0],
        "description": "Identity matrix for RGB video - no transformation applied"
    }
    
    return identity_matrix, params


def get_video_dimensions(video_path: str) -> tuple:
    """
    获取视频的宽度和高度
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        (width, height) 或 None如果获取失败
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    return (width, height)


def find_video_files(root_dir: str) -> List[str]:
    """
    递归查找所有视频文件
    
    Args:
        root_dir: 根目录路径
        
    Returns:
        List[str]: 视频文件路径列表
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    video_files = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in video_extensions:
                video_files.append(file_path)
    
    return sorted(video_files)


def save_identity_matrix_data(matrix: np.ndarray, params: Dict, output_dir: str, video_name: str):
    """
    保存单位矩阵和参数文件
    
    Args:
        matrix: 单位矩阵
        params: 参数字典
        output_dir: 输出目录
        video_name: 视频名称（不含扩展名）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存矩阵文件（使用与透视变换脚本相同的命名规范）
    matrix_path = os.path.join(output_dir, f"{video_name}_perspective_matrix.npy")
    np.save(matrix_path, matrix)
    
    # 保存参数文件
    params_path = os.path.join(output_dir, f"{video_name}_perspective_params.json")
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    print(f"💾 保存单位矩阵数据:")
    print(f"   矩阵: {matrix_path}")
    print(f"   参数: {params_path}")


def process_video_identity_matrix(input_video_path: str, output_dir: str) -> bool:
    """
    为单个视频生成单位矩阵文件
    
    Args:
        input_video_path: 输入视频路径
        output_dir: 输出目录
        
    Returns:
        bool: 是否成功处理
    """
    # 获取视频尺寸信息
    dimensions = get_video_dimensions(input_video_path)
    if dimensions is None:
        return False
    
    width, height = dimensions
    print(f"🎬 处理视频: {os.path.basename(input_video_path)}")
    print(f"   分辨率: {width}x{height}")
    
    # 生成单位矩阵和参数
    identity_matrix, params = create_identity_matrix(width, height)
    
    # 获取视频名称（不含扩展名）
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    
    # 保存矩阵数据
    save_identity_matrix_data(identity_matrix, params, output_dir, video_name)
    
    print(f"✅ 单位矩阵生成完成")
    return True


def process_directory_recursive(input_dir: str, output_dir: str):
    """
    递归处理目录下的所有视频文件，保持完整的目录结构
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
    """
    print(f"🔍 搜索视频文件: {input_dir}")
    video_files = find_video_files(input_dir)
    
    if not video_files:
        print(f"❌ 在 {input_dir} 中未找到视频文件")
        return
    
    print(f"📁 找到 {len(video_files)} 个视频文件:")
    for i, video_file in enumerate(video_files, 1):
        rel_path = os.path.relpath(video_file, input_dir)
        print(f"   {i:3d}. {rel_path}")
    
    print(f"\n🚀 开始生成单位矩阵文件...")
    print(f"📝 目录结构映射:")
    print(f"   输入根目录: {input_dir}")
    print(f"   输出根目录: {output_dir}")
    print(f"   保持相同的子目录结构")
    
    success_count = 0
    fail_count = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"处理 {i}/{len(video_files)}: {os.path.basename(video_file)}")
        print(f"{'='*60}")
        
        # 计算相对路径，保持完整的目录结构
        rel_path = os.path.relpath(video_file, input_dir)
        rel_dir = os.path.dirname(rel_path)
        
        # 在输出目录中重建相同的目录结构
        if rel_dir:
            video_output_dir = os.path.join(output_dir, rel_dir)
        else:
            video_output_dir = output_dir
        
        print(f"📂 目录映射:")
        print(f"   输入: {os.path.dirname(video_file)}")
        print(f"   输出: {video_output_dir}")
        
        success = process_video_identity_matrix(video_file, video_output_dir)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            print(f"❌ 处理失败: {video_file}")
    
    print(f"\n{'='*60}")
    print(f"📊 处理统计:")
    print(f"   总计: {len(video_files)} 个视频")
    print(f"   成功: {success_count} 个")
    print(f"   失败: {fail_count} 个")
    print(f"   成功率: {success_count/len(video_files)*100:.1f}%")
    print(f"📁 输出目录: {output_dir}")
    print(f"✅ 已为所有RGB视频生成单位矩阵文件")
    print(f"{'='*60}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RGB视频单位矩阵生成脚本')
    parser.add_argument('--input_dir', 
                       required=True,
                       help='输入目录路径（包含RGB视频文件的文件夹）')
    parser.add_argument('--output_dir',
                       help='输出目录路径（默认在输入目录下创建identity_matrices子文件夹）')
    
    args = parser.parse_args()
    
    # 验证输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, 'identity_matrices')
    
    # 打印配置信息
    print("="*60)
    print("RGB视频单位矩阵生成")
    print("="*60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"矩阵类型: 3x3单位矩阵")
    print(f"文件命名: 与透视变换脚本保持一致")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开始处理
    process_directory_recursive(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
