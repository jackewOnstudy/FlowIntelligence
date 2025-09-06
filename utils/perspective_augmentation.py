#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态视频透视变换数据增强脚本

功能特性:
- 遍历指定文件夹下的所有视频文件（包括子文件夹）
- 为每个视频生成适度的透视变换（避免过强的缩放）
- 应用透视变换并保存增强后的视频
- 保存透视变换矩阵作为后续标签使用

使用示例:
1. 对指定文件夹进行透视变换增强:
   python perspective_augmentation.py --input_dir /mnt/mDisk2/APIDIS/mm

2. 指定输出文件夹:
   python perspective_augmentation.py --input_dir /mnt/mDisk2/APIDIS/mm --output_dir /path/to/output

3. 设置变换强度:
   python perspective_augmentation.py --input_dir /mnt/mDisk2/APIDIS/mm --max_shift 0.1

4. 查看帮助信息:
   python perspective_augmentation.py --help

输出:
- 增强后的视频文件（保持原始文件名）
- 透视变换矩阵文件（.npy格式）
- 变换参数记录文件（.json格式）
- 完全保持输入目录的结构和命名
"""

import os
import glob
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import random


def generate_perspective_matrix(image_width: int, image_height: int, 
                              max_shift: float = 0.05) -> Tuple[np.ndarray, Dict]:
    """
    生成适度的透视变换矩阵，避免过强的缩放关系
    
    Args:
        image_width: 图像宽度
        image_height: 图像高度  
        max_shift: 最大偏移比例，相对于图像尺寸 (0.05表示最大偏移5%)
        
    Returns:
        perspective_matrix: 3x3透视变换矩阵
        transform_params: 变换参数字典
    """
    # 原始四个角点
    src_points = np.float32([
        [0, 0],                        # 左上角
        [image_width, 0],              # 右上角  
        [image_width, image_height],   # 右下角
        [0, image_height]              # 左下角
    ])
    
    # 计算最大偏移像素
    max_shift_w = int(image_width * max_shift)
    max_shift_h = int(image_height * max_shift)
    
    # 为每个角点生成随机偏移，但保持合理范围
    dst_points = np.float32([
        # 左上角 - 只能向右下偏移
        [random.randint(0, max_shift_w), random.randint(0, max_shift_h)],
        # 右上角 - 只能向左下偏移
        [image_width - random.randint(0, max_shift_w), random.randint(0, max_shift_h)],
        # 右下角 - 只能向左上偏移
        [image_width - random.randint(0, max_shift_w), image_height - random.randint(0, max_shift_h)],
        # 左下角 - 只能向右上偏移
        [random.randint(0, max_shift_w), image_height - random.randint(0, max_shift_h)]
    ])
    
    # 计算透视变换矩阵
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 记录变换参数
    transform_params = {
        "src_points": src_points.tolist(),
        "dst_points": dst_points.tolist(),
        "image_size": [image_width, image_height],
        "max_shift_ratio": max_shift,
        "max_shift_pixels": [max_shift_w, max_shift_h]
    }
    
    return perspective_matrix, transform_params


def apply_perspective_transform_to_video(input_video_path: str, output_video_path: str, 
                                       perspective_matrix: np.ndarray) -> bool:
    """
    对视频应用透视变换
    
    Args:
        input_video_path: 输入视频路径
        output_video_path: 输出视频路径
        perspective_matrix: 透视变换矩阵
        
    Returns:
        bool: 是否成功处理
    """
    # 打开输入视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {input_video_path}")
        return False
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"❌ 无法创建输出视频: {output_video_path}")
        cap.release()
        return False
    
    print(f"🎬 处理视频: {os.path.basename(input_video_path)}")
    print(f"   分辨率: {width}x{height}, 帧率: {fps:.2f}, 总帧数: {total_frames}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 应用透视变换
        transformed_frame = cv2.warpPerspective(frame, perspective_matrix, (width, height))
        
        # 写入输出视频
        out.write(transformed_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"   进度: {progress:.1f}% ({frame_count}/{total_frames})")
    
    # 释放资源
    cap.release()
    out.release()
    
    print(f"✅ 视频处理完成: {output_video_path}")
    return True


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


def save_transform_data(matrix: np.ndarray, params: Dict, output_dir: str, video_name: str):
    """
    保存透视变换矩阵和参数
    
    Args:
        matrix: 透视变换矩阵
        params: 变换参数
        output_dir: 输出目录
        video_name: 视频名称（不含扩展名）
    """
    # 保存矩阵
    matrix_path = os.path.join(output_dir, f"{video_name}_perspective_matrix.npy")
    np.save(matrix_path, matrix)
    
    # 保存参数
    params_path = os.path.join(output_dir, f"{video_name}_perspective_params.json")
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    
    print(f"💾 保存变换数据:")
    print(f"   矩阵: {matrix_path}")
    print(f"   参数: {params_path}")


def process_video_perspective_augmentation(input_video_path: str, output_dir: str, 
                                         max_shift: float = 0.05) -> bool:
    """
    对单个视频进行透视变换增强
    
    Args:
        input_video_path: 输入视频路径
        output_dir: 输出目录
        max_shift: 最大偏移比例
        
    Returns:
        bool: 是否成功处理
    """
    # 获取视频信息
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {input_video_path}")
        return False
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # 生成透视变换矩阵
    perspective_matrix, transform_params = generate_perspective_matrix(width, height, max_shift)
    
    # 构造输出路径 - 保持原始文件名和扩展名
    original_filename = os.path.basename(input_video_path)
    video_name = os.path.splitext(original_filename)[0]
    video_ext = os.path.splitext(original_filename)[1]
    
    # 输出视频保持原始文件名（不添加后缀）
    output_video_path = os.path.join(output_dir, original_filename)
    
    # 应用透视变换
    success = apply_perspective_transform_to_video(input_video_path, output_video_path, perspective_matrix)
    
    if success:
        # 保存变换数据
        save_transform_data(perspective_matrix, transform_params, output_dir, video_name)
    
    return success


def process_directory_recursive(input_dir: str, output_dir: str, max_shift: float = 0.05):
    """
    递归处理目录下的所有视频文件，保持完整的目录结构和文件命名一致性
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        max_shift: 最大偏移比例
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
    
    print(f"\n🚀 开始处理视频文件...")
    print(f"📝 目录结构映射:")
    print(f"   输入根目录: {input_dir}")
    print(f"   输出根目录: {output_dir}")
    print(f"   保持相同的子目录结构和文件名")
    
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
        print(f"   文件: {os.path.basename(video_file)} → {os.path.basename(video_file)}")
        
        success = process_video_perspective_augmentation(video_file, video_output_dir, max_shift)
        
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
    print(f"✅ 目录结构和文件命名保持与输入一致")
    print(f"{'='*60}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多模态视频透视变换数据增强脚本')
    parser.add_argument('--input_dir', 
                       required=True,
                       help='输入目录路径（包含视频文件的文件夹）')
    parser.add_argument('--output_dir',
                       help='输出目录路径（默认在输入目录下创建perspective_augmented子文件夹）')
    parser.add_argument('--max_shift',
                       type=float,
                       default=0.05,
                       help='最大偏移比例，相对于图像尺寸 (默认: 0.05，即5%%)')
    parser.add_argument('--seed',
                       type=int,
                       help='随机种子，用于可重现的结果')
    
    args = parser.parse_args()
    
    # 验证输入目录
    if not os.path.exists(args.input_dir):
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, 'perspective_augmented')
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"🎲 设置随机种子: {args.seed}")
    
    # 验证参数
    if not (0.01 <= args.max_shift <= 0.2):
        print(f"⚠️ 警告: max_shift={args.max_shift} 可能不合适，建议范围: 0.01-0.2")
    
    # 打印配置信息
    print("="*60)
    print("多模态视频透视变换数据增强")
    print("="*60)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"最大偏移: {args.max_shift*100:.1f}%")
    print(f"随机种子: {args.seed if args.seed else '未设置'}")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开始处理
    process_directory_recursive(args.input_dir, args.output_dir, args.max_shift)


if __name__ == "__main__":
    main()
