#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算匹配模态对之间的相对透视变换关系脚本

功能特性:
- 从透视变换矩阵计算匹配模态对之间的相对变换关系
- 自动扫描匹配结果目录，识别所有模态匹配对
- 为每个匹配对计算并保存相对透视变换矩阵作为标签
- 支持RGB和多模态数据的统一处理

数学原理:
假设原始视角为I，两个模态经过透视变换后：
- modality1 = I × H1  
- modality2 = I × H2
相对变换关系为: H_relative = H2 × H1^(-1)
这样 modality1 经过 H_relative 变换后应该与 modality2 对齐

使用示例:
1. 处理所有匹配结果:
   python calculate_relative_transforms.py --rgb_dir /mnt/mDisk2/APIDIS_P/mp4 --mm_dir /mnt/mDisk2/APIDIS_P/mm --output_dir /mnt/mDisk2/APIDIS_P/MultiModal_Output

2. 处理指定场景:
   python calculate_relative_transforms.py --rgb_dir /mnt/mDisk2/APIDIS_P/mp4 --mm_dir /mnt/mDisk2/APIDIS_P/mm --output_dir /mnt/mDisk2/APIDIS_P/MultiModal_Output --scenes A1 A2

3. 查看帮助信息:
   python calculate_relative_transforms.py --help

输出:
- relative_transform_matrix.npy: 相对透视变换矩阵
- relative_transform_params.json: 相对变换参数和元数据
"""

import os
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob


def load_perspective_matrix(matrix_file: str) -> Optional[np.ndarray]:
    """
    加载透视变换矩阵文件
    
    Args:
        matrix_file: 矩阵文件路径
        
    Returns:
        3x3透视变换矩阵，如果加载失败返回None
    """
    try:
        matrix = np.load(matrix_file)
        if matrix.shape != (3, 3):
            print(f"❌ 矩阵维度错误: {matrix.shape}, 期望 (3, 3)")
            return None
        return matrix
    except Exception as e:
        print(f"❌ 加载矩阵文件失败: {matrix_file} - {e}")
        return None


def calculate_relative_transform(matrix1: np.ndarray, matrix2: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    计算两个透视变换矩阵之间的相对变换关系
    
    Args:
        matrix1: 第一个模态的透视变换矩阵 H1
        matrix2: 第二个模态的透视变换矩阵 H2
        
    Returns:
        relative_matrix: 相对透视变换矩阵 H_relative = H2 × H1^(-1)
        params: 相对变换参数信息
    """
    try:
        # 计算H1的逆矩阵
        matrix1_inv = np.linalg.inv(matrix1)
        
        # 计算相对变换矩阵: H_relative = H2 × H1^(-1)
        relative_matrix = np.dot(matrix2, matrix1_inv)
        
        # 计算变换的"距离"度量（Frobenius范数）
        transform_distance = np.linalg.norm(relative_matrix - np.eye(3), 'fro')
        
        # 计算矩阵的条件数（数值稳定性指标）
        condition_number1 = np.linalg.cond(matrix1)
        condition_number2 = np.linalg.cond(matrix2)
        condition_number_relative = np.linalg.cond(relative_matrix)
        
        params = {
            "matrix1_condition": float(condition_number1),
            "matrix2_condition": float(condition_number2),
            "relative_condition": float(condition_number_relative),
            "transform_distance": float(transform_distance),
            "matrix1_determinant": float(np.linalg.det(matrix1)),
            "matrix2_determinant": float(np.linalg.det(matrix2)),
            "relative_determinant": float(np.linalg.det(relative_matrix)),
            "computation_method": "H_relative = H2 × H1^(-1)",
            "description": "Relative perspective transform from modality1 to modality2"
        }
        
        return relative_matrix, params
        
    except np.linalg.LinAlgError as e:
        print(f"❌ 矩阵计算错误: {e}")
        return None, None
    except Exception as e:
        print(f"❌ 计算相对变换失败: {e}")
        return None, None


def find_matrix_file(search_dir: str, video_name: str) -> Optional[str]:
    """
    在指定目录中查找对应的透视变换矩阵文件
    
    Args:
        search_dir: 搜索目录
        video_name: 视频名称（不含扩展名）
        
    Returns:
        矩阵文件路径，如果未找到返回None
    """
    # 尝试不同的命名模式
    possible_names = [
        f"{video_name}_perspective_matrix.npy",
        f"{video_name.split('_')[0]}_perspective_matrix.npy"  # 处理A1_frequency_domain -> A1的情况
    ]
    
    for name in possible_names:
        matrix_path = os.path.join(search_dir, name)
        if os.path.exists(matrix_path):
            return matrix_path
    
    # 递归搜索子目录
    for root, dirs, files in os.walk(search_dir):
        for name in possible_names:
            if name in files:
                return os.path.join(root, name)
    
    return None


def get_modality_matrix_path(modality: str, scene: str, rgb_dir: str, mm_dir: str) -> Optional[str]:
    """
    获取指定模态的透视变换矩阵路径
    
    Args:
        modality: 模态名称 (RGB, FreqDomain, MotionThermal等)
        scene: 场景名称 (A1, A2等)
        rgb_dir: RGB视频目录
        mm_dir: 多模态视频目录
        
    Returns:
        矩阵文件路径，如果未找到返回None
    """
    if modality == "RGB":
        # RGB视频的矩阵在rgb_dir下
        return find_matrix_file(rgb_dir, scene)
    else:
        # 多模态视频的矩阵在对应场景的子目录下
        scene_dir = os.path.join(mm_dir, scene)
        if not os.path.exists(scene_dir):
            return None
        
        # 根据模态名称映射到文件名
        modality_mapping = {
            "FreqDomain": "frequency_domain",
            "MotionThermal": "motion_thermal", 
            "OpticalFlow": "optical_flow",
            "TempGradient": "temporal_gradient",
            "TextureRemoval": "texture_removal"
        }
        
        if modality in modality_mapping:
            video_name = f"{scene}_{modality_mapping[modality]}"
            return find_matrix_file(scene_dir, video_name)
        else:
            print(f"⚠️ 未知模态: {modality}")
            return None


def scan_match_results(output_dir: str, scenes: List[str] = None) -> List[Tuple[str, str, str]]:
    """
    扫描匹配结果目录，获取所有场景和模态对信息
    
    Args:
        output_dir: 匹配结果输出目录
        scenes: 指定要处理的场景列表，None表示处理所有场景
        
    Returns:
        [(scene, modality1, modality2), ...] 的列表
    """
    match_pairs = []
    
    if not os.path.exists(output_dir):
        print(f"❌ 输出目录不存在: {output_dir}")
        return match_pairs
    
    # 扫描所有场景目录
    for scene_item in os.listdir(output_dir):
        scene_path = os.path.join(output_dir, scene_item)
        if not os.path.isdir(scene_path):
            continue
            
        # 如果指定了场景列表，只处理指定的场景
        if scenes is not None and scene_item not in scenes:
            continue
        
        # 扫描场景目录下的模态匹配对文件夹
        for match_item in os.listdir(scene_path):
            match_path = os.path.join(scene_path, match_item)
            if not os.path.isdir(match_path):
                continue
                
            # 解析模态匹配对名称 (例如: FreqDomain_vs_MotionThermal)
            if "_vs_" in match_item:
                modality1, modality2 = match_item.split("_vs_", 1)
                match_pairs.append((scene_item, modality1, modality2))
    
    return sorted(match_pairs)


def save_relative_transform(matrix: np.ndarray, params: Dict, output_path: str, 
                          scene: str, modality1: str, modality2: str):
    """
    保存相对透视变换矩阵和参数
    
    Args:
        matrix: 相对透视变换矩阵
        params: 参数字典
        output_path: 输出目录路径
        scene: 场景名称
        modality1: 第一个模态名称  
        modality2: 第二个模态名称
    """
    os.makedirs(output_path, exist_ok=True)
    
    # 保存矩阵文件
    matrix_file = os.path.join(output_path, "relative_transform_matrix.npy")
    np.save(matrix_file, matrix)
    
    # 增强参数信息
    enhanced_params = {
        "scene": scene,
        "modality1": modality1,
        "modality2": modality2,
        "relative_transform_matrix_shape": matrix.shape,
        "generation_timestamp": str(np.datetime64('now')),
        **params
    }
    
    # 保存参数文件
    params_file = os.path.join(output_path, "relative_transform_params.json")
    with open(params_file, 'w', encoding='utf-8') as f:
        json.dump(enhanced_params, f, indent=2, ensure_ascii=False)
    
    print(f"💾 保存相对变换数据:")
    print(f"   矩阵: {matrix_file}")
    print(f"   参数: {params_file}")


def process_match_pair(scene: str, modality1: str, modality2: str, 
                      rgb_dir: str, mm_dir: str, output_dir: str) -> bool:
    """
    处理单个匹配对，计算并保存相对透视变换
    
    Args:
        scene: 场景名称
        modality1: 第一个模态
        modality2: 第二个模态
        rgb_dir: RGB视频目录
        mm_dir: 多模态视频目录
        output_dir: 输出目录
        
    Returns:
        bool: 是否成功处理
    """
    print(f"📐 计算相对变换: {modality1} -> {modality2}")
    
    # 获取两个模态的透视变换矩阵路径
    matrix1_path = get_modality_matrix_path(modality1, scene, rgb_dir, mm_dir)
    matrix2_path = get_modality_matrix_path(modality2, scene, rgb_dir, mm_dir)
    
    if matrix1_path is None:
        print(f"❌ 未找到 {modality1} 的透视变换矩阵文件")
        return False
        
    if matrix2_path is None:
        print(f"❌ 未找到 {modality2} 的透视变换矩阵文件")
        return False
    
    print(f"   矩阵1: {matrix1_path}")
    print(f"   矩阵2: {matrix2_path}")
    
    # 加载透视变换矩阵
    matrix1 = load_perspective_matrix(matrix1_path)
    matrix2 = load_perspective_matrix(matrix2_path)
    
    if matrix1 is None or matrix2 is None:
        print("❌ 矩阵加载失败")
        return False
    
    # 计算相对变换
    relative_matrix, params = calculate_relative_transform(matrix1, matrix2)
    
    if relative_matrix is None:
        print("❌ 相对变换计算失败")
        return False
    
    # 保存结果
    match_output_path = os.path.join(output_dir, scene, f"{modality1}_vs_{modality2}")
    save_relative_transform(relative_matrix, params, match_output_path, scene, modality1, modality2)
    
    print(f"✅ 相对变换计算完成")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='计算匹配模态对之间的相对透视变换关系')
    parser.add_argument('--rgb_dir', 
                       required=True,
                       help='RGB视频目录路径')
    parser.add_argument('--mm_dir',
                       required=True, 
                       help='多模态视频目录路径')
    parser.add_argument('--output_dir',
                       required=True,
                       help='匹配结果输出目录路径')
    parser.add_argument('--scenes',
                       nargs='*',
                       help='指定要处理的场景列表，例如: A1 A2 A3')
    
    args = parser.parse_args()
    
    # 验证输入目录
    for dir_path, dir_name in [(args.rgb_dir, 'RGB目录'), 
                               (args.mm_dir, '多模态目录'), 
                               (args.output_dir, '输出目录')]:
        if not os.path.exists(dir_path):
            print(f"❌ {dir_name}不存在: {dir_path}")
            return
    
    # 打印配置信息
    print("="*60)
    print("相对透视变换关系计算")
    print("="*60)
    print(f"RGB视频目录: {args.rgb_dir}")
    print(f"多模态目录: {args.mm_dir}")
    print(f"输出目录: {args.output_dir}")
    if args.scenes:
        print(f"指定场景: {args.scenes}")
    else:
        print("处理所有场景")
    print("="*60)
    
    # 扫描匹配结果
    match_pairs = scan_match_results(args.output_dir, args.scenes)
    
    if not match_pairs:
        print("❌ 未找到任何匹配对")
        return
    
    print(f"📁 找到 {len(match_pairs)} 个匹配对:")
    for i, (scene, mod1, mod2) in enumerate(match_pairs, 1):
        print(f"   {i:3d}. {scene}: {mod1} vs {mod2}")
    
    print(f"\n🚀 开始计算相对透视变换...")
    
    success_count = 0
    fail_count = 0
    
    for i, (scene, modality1, modality2) in enumerate(match_pairs, 1):
        print(f"\n{'='*60}")
        print(f"处理 {i}/{len(match_pairs)}: {scene} - {modality1} vs {modality2}")
        print(f"{'='*60}")
        
        success = process_match_pair(scene, modality1, modality2, 
                                   args.rgb_dir, args.mm_dir, args.output_dir)
        
        if success:
            success_count += 1
        else:
            fail_count += 1
            print(f"❌ 处理失败: {scene} - {modality1} vs {modality2}")
    
    print(f"\n{'='*60}")
    print(f"📊 处理统计:")
    print(f"   总计: {len(match_pairs)} 个匹配对")
    print(f"   成功: {success_count} 个")
    print(f"   失败: {fail_count} 个")
    print(f"   成功率: {success_count/len(match_pairs)*100:.1f}%")
    print(f"✅ 所有相对透视变换关系已计算完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
