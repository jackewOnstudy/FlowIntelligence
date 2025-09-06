#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态匹配结果准确率与匹配数统计脚本
分析 MultiModal_Output 目录下的匹配结果

功能特性:
- 计算匹配准确率（支持多个像素误差阈值）
- 统计RANSAC算法剔除后的有效匹配数
- 计算匹配保留率（有效匹配数/总匹配数）
- 支持排除特定模态
- 生成详细的Excel报告

使用示例:
1. 分析所有模态（传统方法，假设视角一致）:
   python analyze_match_results.py

2. 使用透视变换标签分析（用于透视变换增强的数据）:
   python analyze_match_results.py --use_perspective_label

3. 排除OpticalFlow模态并使用透视变换标签:
   python analyze_match_results.py --exclude_modalities OpticalFlow --use_perspective_label

4. 排除多个模态:
   python analyze_match_results.py --exclude_modalities OpticalFlow FreqDomain

5. 指定自定义输出目录和Excel文件名:
   python analyze_match_results.py --output_dir /path/to/output --excel_output custom_results.xlsx --use_perspective_label

6. 查看帮助信息:
   python analyze_match_results.py --help

输出Excel文件包含的工作表:
- 详细结果_准确率: 每个场景和模态的准确率详细数据
- 详细结果_匹配数: 每个场景和模态的匹配数详细数据（包含保留率）
- 场景平均准确率: 每个场景的平均准确率
- 模态平均准确率: 每个模态的平均准确率
- 全局平均准确率: 所有数据的全局平均准确率
- 场景平均匹配数: 每个场景的平均匹配数
- 模态平均匹配数: 每个模态的平均匹配数
- 全局平均匹配数: 所有数据的全局平均匹配数
"""

import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import json
import cv2

use_pt_level = False
video_root = "/mnt/mDisk2/APIDIS_P/mp4"

def get_video_resolution_from_match_path(match_path: str, video_root: str = video_root) -> tuple:
    """
    从匹配结果路径中提取场景名，读取对应视频的分辨率。
    
    Args:
        match_path: 匹配结果文件路径，例如 MultiModal_Output_ransac_5_05/K1/...
        video_root: 视频根目录，默认 video_root
    
    Returns:
        (width, height) 分辨率元组；如果找不到视频或读取失败，返回 None
    """
    # 1. 提取场景名
    parts = os.path.normpath(match_path).split(os.sep)
    try:
        k_index = parts.index('MultiModal_Output') + 1
        scene_name = parts[k_index]  # 比如 "K1"
    except (ValueError, IndexError):
        print(f"❌ 无法从路径中提取场景名: {match_path}")
        return None

    # 2. 构造视频路径
    video_file = os.path.join(video_root, f"{scene_name}.mp4")
    if not os.path.exists(video_file):
        print(f"❌ 视频文件未找到: {video_file}")
        return None

    # 3. 使用 OpenCV 获取视频分辨率
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_file}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"🎥 视频 {scene_name}.mp4 分辨率: {width} x {height}")
    return (width, height)


def patch_id_to_coords(patch_id: int, patch_size: int, image_width: int) -> Tuple[float, float]:
    """
    将 patch ID 转换为图像坐标（中心点）
    - patch_size: 8, 16, 32, ...
    - image_width: 图像宽度（像素）
    """
    cols_per_row = image_width // patch_size
    row = patch_id // cols_per_row
    col = patch_id % cols_per_row
    x = col * patch_size + patch_size / 2
    y = row * patch_size + patch_size / 2
    return (x, y)

def load_match_ids_from_txt(file_path: str) -> List[Tuple[int, int]]:
    """
    读取匹配txt文件，格式如下：
    - 第一行是匹配数量（整数）
    - 后续每行有3个数字，用空格分隔，格式如 id1 id2 value
    返回只提取 (id1, id2) 的列表
    """
    matches = []
    print(f"📄 读取匹配文件: {file_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            return matches
        
        try:
            total_matches = int(lines[0].strip())  # 读取第一行匹配数量
        except:
            total_matches = None  # 如果第一行不是整数也不报错，忽略
        
        for line in lines[1:]:  # 跳过第一行
            try:
                parts = line.strip().split()  # 默认按空白分割
                if len(parts) < 2:
                    continue
                id1, id2 = int(parts[0]), int(parts[1])
                matches.append((id1, id2))
            except:
                continue
    
    return matches


def load_relative_transform_matrix(match_file_path: str) -> np.ndarray:
    """
    加载相对透视变换矩阵作为标签
    
    Args:
        match_file_path: 匹配文件路径
        
    Returns:
        3x3相对透视变换矩阵，如果未找到或加载失败返回None
    """
    # 从匹配文件路径推断相对变换矩阵路径
    # 匹配文件路径例如: .../A1/FreqDomain_vs_MotionThermal/MatchResult/List/.../8x8.txt
    # 相对变换矩阵路径: .../A1/FreqDomain_vs_MotionThermal/relative_transform_matrix.npy
    
    # 找到包含 "_vs_" 的目录层级
    current_path = match_file_path
    modality_pair_dir = None
    
    while current_path and current_path != os.path.dirname(current_path):
        current_path = os.path.dirname(current_path)
        dir_name = os.path.basename(current_path)
        
        if '_vs_' in dir_name:
            modality_pair_dir = current_path
            break
    
    if not modality_pair_dir:
        print(f"❌ 无法找到包含'_vs_'的目录层级: {match_file_path}")
        return None
        
    matrix_path = os.path.join(modality_pair_dir, 'relative_transform_matrix.npy')
    
    try:
        if os.path.exists(matrix_path):
            matrix = np.load(matrix_path)
            if matrix.shape == (3, 3):
                print(f"✅ 加载相对变换矩阵: {matrix_path}")
                return matrix
            else:
                print(f"❌ 相对变换矩阵维度错误: {matrix.shape}")
                return None
        else:
            print(f"⚠️ 相对变换矩阵文件未找到: {matrix_path}")
            return None
    except Exception as e:
        print(f"❌ 加载相对变换矩阵失败: {matrix_path} - {e}")
        return None


def calculate_accuracy(match_file_path: str, use_perspective_label: bool = False) -> Dict[str, any]:
    """
    使用RANSAC评估匹配准确率，并统计剔除后的匹配数。
    支持使用透视变换标签矩阵计算准确性。

    Args:
        match_file_path: 匹配文件路径
        use_perspective_label: 是否使用透视变换标签矩阵计算准确性

    Returns:
        Dict containing:
        - 'accuracy': Dict[int, float] - key 是像素误差阈值，value 是该阈值下的准确率
        - 'match_count': int - RANSAC算法剔除后的有效匹配数
        - 'total_matches': int - 原始总匹配数
        - 'use_perspective_label': bool - 是否使用了透视变换标签
        - 'perspective_available': bool - 透视变换标签是否可用
    """
    # 根据 use_pt_level 确定要使用的阈值
    if use_pt_level:
        thresholds = [4, 8, 12, 16, 20]
    else:
        thresholds = list(range(1, 11))
    
    # 错误情况下返回的默认准确率字典
    default_accuracy = {k: 0.0 for k in thresholds}
    
    # 尝试加载相对透视变换矩阵
    relative_transform_matrix = None
    perspective_available = False
    if use_perspective_label:
        relative_transform_matrix = load_relative_transform_matrix(match_file_path)
        perspective_available = relative_transform_matrix is not None
        if not perspective_available:
            print(f"⚠️ 启用透视变换标签但未找到矩阵文件，将使用传统方法")
    
    try:
        matches = load_match_ids_from_txt(match_file_path)
        total_matches = len(matches)
        
        if not matches or len(matches) < 1:
            print(f"❌ 匹配数据为空或无效: {matches}")
            return {
                'accuracy': default_accuracy,
                'match_count': 0,
                'total_matches': 0,
                'use_perspective_label': use_perspective_label,
                'perspective_available': perspective_available
            }

        filename = os.path.basename(match_file_path)
        if 'x' not in filename:
            print(f"❌ 文件名无法识别 patch size: {filename}")
            return {
                'accuracy': default_accuracy,
                'match_count': 0,
                'total_matches': total_matches,
                'use_perspective_label': use_perspective_label,
                'perspective_available': perspective_available
            }

        patch_size = int(filename.split('x')[0])
        res = get_video_resolution_from_match_path(match_file_path)
        if res is None:
            print(f"❌ 无法获取视频分辨率: {match_file_path}")
            return {
                'accuracy': default_accuracy,
                'match_count': 0,
                'total_matches': total_matches,
                'use_perspective_label': use_perspective_label,
                'perspective_available': perspective_available
            }
        width, height = res

        # 计算真实坐标
        src_pts = np.array([patch_id_to_coords(m[0], patch_size, width) for m in matches])
        dst_pts = np.array([patch_id_to_coords(m[1], patch_size, width) for m in matches])

        # 检查匹配点数量是否足够进行RANSAC
        if len(matches) < 4:
            print(f"⚠️ 匹配点数量不足({len(matches)}<4)，跳过RANSAC，直接计算准确率: {match_file_path}")
            # 直接使用所有匹配点
            inlier_src = src_pts
            inlier_dst = dst_pts
            match_count = len(src_pts)
        else:
            ransac_threshold = max(3.0, patch_size * 0.5)  # 动态阈值
            max_iters = 2000  # 最大迭代次数
            confidence = 0.95  # 置信度

            # 计算单应性矩阵
            H, mask = cv2.findHomography(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_threshold,
                maxIters=max_iters,
                confidence=confidence
            )

            # # 使用单位矩阵，即不进行任何变换
            # transformed_pts = src_pts  # 因为 H = I

            if mask is None:
                print(f"⚠️ RANSAC 返回无效 mask，使用所有匹配点计算准确率: {match_file_path}")
                # 当RANSAC失败时，使用所有原始匹配点（不进行过滤）
                inlier_src = src_pts
                inlier_dst = dst_pts
                match_count = len(src_pts)  # 所有匹配点都算作有效匹配
            else:
                # 提取内点并计算有效匹配数
                inlier_mask = mask.ravel() == 1
                inlier_src = src_pts[inlier_mask]
                inlier_dst = dst_pts[inlier_mask]
                match_count = len(inlier_src)  # RANSAC剔除后的有效匹配数

        # 计算误差
        if use_perspective_label and perspective_available:
            # 使用透视变换标签计算误差
            print(f"📐 使用透视变换标签计算准确性")
            # 将第一个模态的点通过相对变换矩阵变换到第二个模态
            inlier_src_homogeneous = np.hstack([inlier_src, np.ones((len(inlier_src), 1))])
            transformed_pts = (relative_transform_matrix @ inlier_src_homogeneous.T).T
            # 归一化齐次坐标
            transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2:]
            
            if not use_pt_level:
                # 欧几里得误差：变换后的点与目标点的距离
                errors = np.linalg.norm(transformed_pts - inlier_dst, axis=1)
            else:
                # 最大值误差 (max of dx, dy)
                dx = np.abs(transformed_pts[:, 0] - inlier_dst[:, 0])
                dy = np.abs(transformed_pts[:, 1] - inlier_dst[:, 1])
                errors = np.maximum(dx, dy)
        else:
            # 传统方法：假设视角一致，直接比较对应点
            if not use_pt_level:
                # 欧几里得误差
                errors = np.linalg.norm(inlier_src - inlier_dst, axis=1)
            else:
                # 最大值误差 (max of dx, dy)
                dx = np.abs(inlier_src[:, 0] - inlier_dst[:, 0])
                dy = np.abs(inlier_src[:, 1] - inlier_dst[:, 1])
                errors = np.maximum(dx, dy)

        total = len(errors)
        acc_by_thresh = {}
        for thresh in thresholds:
            inliers = np.sum(errors <= thresh)
            acc_by_thresh[thresh] = inliers / total if total > 0 else 0.0

        if sum(acc_by_thresh.values()) == 0:
            # 正确匹配过少，计算的ransac 透视矩阵不对，筛选出来的inlier point 其实是错误的
            print(f"❌ 所有阈值下的准确率都为0: {acc_by_thresh}; Match count: {match_count}; Total matches: {total_matches}")
            inlier_src = src_pts
            inlier_dst = dst_pts
            match_count = len(src_pts)

            # 重新计算误差，这次也要考虑透视变换标签
            if use_perspective_label and perspective_available:
                # 使用透视变换标签计算误差
                print(f"📐 使用透视变换标签重新计算准确性")
                inlier_src_homogeneous = np.hstack([inlier_src, np.ones((len(inlier_src), 1))])
                transformed_pts = (relative_transform_matrix @ inlier_src_homogeneous.T).T
                transformed_pts = transformed_pts[:, :2] / transformed_pts[:, 2:]
                
                if not use_pt_level:
                    errors = np.linalg.norm(transformed_pts - inlier_dst, axis=1)
                else:
                    dx = np.abs(transformed_pts[:, 0] - inlier_dst[:, 0])
                    dy = np.abs(transformed_pts[:, 1] - inlier_dst[:, 1])
                    errors = np.maximum(dx, dy)
            else:
                if not use_pt_level:
                    errors = np.linalg.norm(inlier_src - inlier_dst, axis=1)
                else:
                    dx = np.abs(inlier_src[:, 0] - inlier_dst[:, 0])
                    dy = np.abs(inlier_src[:, 1] - inlier_dst[:, 1])
                    errors = np.maximum(dx, dy)

            total = len(errors)
            acc_by_thresh = {}
            for thresh in thresholds:
                inliers = np.sum(errors <= thresh)
                acc_by_thresh[thresh] = inliers / total if total > 0 else 0.0


        return {
            'accuracy': acc_by_thresh,
            'match_count': match_count,
            'total_matches': total_matches,
            'use_perspective_label': use_perspective_label,
            'perspective_available': perspective_available
        }

    except Exception as e:
        print(f"❌ 处理失败: {match_file_path} - {str(e)}")
        return {
            'accuracy': default_accuracy,
            'match_count': 0,
            'total_matches': 0,
            'use_perspective_label': use_perspective_label,
            'perspective_available': perspective_available
        }


class MatchResultAnalyzer:
    def __init__(self, output_dir: str, exclude_modalities: List[str] = None, scene_prefix: str = 'V', 
                 use_perspective_label: bool = False):
        """
        初始化分析器
        
        Args:
            output_dir: MultiModal_Output_ransac_5_05 目录路径
            exclude_modalities: 要排除的模态列表，例如 ['OpticalFlow', 'FreqDomain']
            scene_prefix: 场景文件夹前缀，例如 'V', 'A', 'K' 等
            use_perspective_label: 是否使用透视变换标签计算准确性
        """
        self.output_dir = output_dir
        self.exclude_modalities = exclude_modalities or []
        self.scene_prefix = scene_prefix
        self.use_perspective_label = use_perspective_label
        self.scenarios = []
        self.modality_pairs = []
        self.grid_sizes = ['8x8', '16x16', '32x32', '64x64']
        
        # 存储所有统计结果
        self.scenario_modality_results = defaultdict(lambda: defaultdict(dict))  # scenario -> modality_pair -> grid_size -> result_dict
        self.scenario_averages = defaultdict(dict)  # scenario -> grid_size -> average_accuracy
        self.modality_averages = defaultdict(dict)  # modality_pair -> grid_size -> average_accuracy
        self.global_averages = {}  # grid_size -> average_accuracy
        
        # 新增：存储匹配数统计结果
        self.scenario_match_counts = defaultdict(lambda: defaultdict(dict))  # scenario -> modality_pair -> grid_size -> match_count
        self.scenario_match_averages = defaultdict(dict)  # scenario -> grid_size -> average_match_count
        self.modality_match_averages = defaultdict(dict)  # modality_pair -> grid_size -> average_match_count
        self.global_match_averages = {}  # grid_size -> average_match_count
        
    def _should_exclude_modality_pair(self, modality_pair: str) -> bool:
        """
        检查模态配对是否应该被排除
        
        Args:
            modality_pair: 模态配对名称，如 'FreqDomain_vs_OpticalFlow'
            
        Returns:
            True 如果应该排除，False 如果应该保留
        """
        if not self.exclude_modalities:
            return False
        
        # 检查模态配对中是否包含任何需要排除的模态
        for exclude_modality in self.exclude_modalities:
            if exclude_modality in modality_pair:
                return True
        
        return False

    def scan_directory_structure(self):
        """扫描目录结构，获取所有场景和模态配对"""
        self.scenarios = []
        self.modality_pairs = set()
        
        for item in os.listdir(self.output_dir):
            if os.path.isdir(os.path.join(self.output_dir, item)) and item.startswith(self.scene_prefix):
                self.scenarios.append(item)
                
                # 扫描该场景下的模态配对
                scenario_dir = os.path.join(self.output_dir, item)
                for modality_item in os.listdir(scenario_dir):
                    modality_path = os.path.join(scenario_dir, modality_item)
                    if os.path.isdir(modality_path) and '_vs_' in modality_item:
                        # 检查是否需要排除该模态配对
                        if not self._should_exclude_modality_pair(modality_item):
                            self.modality_pairs.add(modality_item)
        
        self.scenarios.sort(key=lambda x: int(x[1:]))  # 按数字排序
        self.modality_pairs = sorted(list(self.modality_pairs))
        
        print(f"发现 {len(self.scenarios)} 个场景: {self.scenarios}")
        print(f"发现 {len(self.modality_pairs)} 个模态配对: {self.modality_pairs}")
        
        if self.exclude_modalities:
            print(f"已排除包含以下模态的配对: {self.exclude_modalities}")
    
    def find_match_files(self, scenario: str, modality_pair: str) -> Dict[str, str]:
        """
        找到指定场景和模态配对的匹配结果文件
        
        Args:
            scenario: 场景名称，如 'K1'
            modality_pair: 模态配对名称，如 'FreqDomain_vs_MotionThermal'
            
        Returns:
            字典，键为网格大小，值为文件路径
        """
        match_files = {}
        
        # 构建路径：scenario/modality_pair/MatchResult/List/子文件夹/
        base_path = os.path.join(self.output_dir, scenario, modality_pair, 'MatchResult', 'List')
        
        if not os.path.exists(base_path):
            return match_files
        
        # 找到List下的子文件夹（无意义的层级文件夹）
        try:
            subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            if not subdirs:
                return match_files
            
            # 取第一个子文件夹
            subdir = subdirs[0]
            match_dir = os.path.join(base_path, subdir)
            
            # 寻找网格大小文件
            for grid_size in self.grid_sizes:
                file_path = os.path.join(match_dir, f"{grid_size}.txt")
                if os.path.exists(file_path):
                    match_files[grid_size] = file_path
                    
        except Exception as e:
            print(f"Error scanning {base_path}: {e}")
        
        return match_files
    
    def analyze_scenario_modality(self, scenario: str, modality_pair: str):
        """分析特定场景和模态配对的结果"""
        match_files = self.find_match_files(scenario, modality_pair)
        
        for grid_size, file_path in match_files.items():
            result_dict = calculate_accuracy(file_path, self.use_perspective_label)
            
            # 存储完整结果
            self.scenario_modality_results[scenario][modality_pair][grid_size] = result_dict
            
            # 单独存储匹配数信息
            self.scenario_match_counts[scenario][modality_pair][grid_size] = {
                'match_count': result_dict['match_count'],
                'total_matches': result_dict['total_matches'],
                'retention_rate': result_dict['match_count'] / result_dict['total_matches'] if result_dict['total_matches'] > 0 else 0.0
            }
            
            print(f"  {scenario} - {modality_pair} - {grid_size}:")
            print(f"    准确率: {result_dict['accuracy']}")
            print(f"    匹配数: {result_dict['match_count']}/{result_dict['total_matches']} "
                  f"(保留率: {self.scenario_match_counts[scenario][modality_pair][grid_size]['retention_rate']:.2%})")
    
    def calculate_scenario_averages(self):
        """计算每个场景所有模态匹配结果的不同网格大小平均准确率"""
        for scenario in self.scenarios:
            for grid_size in self.grid_sizes:
                acc_by_thresh = defaultdict(list)  # 每个像素阈值 -> 所有模态准确率列表
                match_counts = []  # 匹配数列表

                for modality_pair in self.modality_pairs:
                    result_dict = self.scenario_modality_results.get(scenario, {}).get(modality_pair, {}).get(grid_size)
                    if result_dict and 'accuracy' in result_dict:
                        acc_dict = result_dict['accuracy']
                        for thresh, acc in acc_dict.items():
                            acc_by_thresh[thresh].append(acc)
                        
                        # 收集匹配数
                        match_counts.append(result_dict['match_count'])

                # 计算每个像素阈值的平均准确率
                if acc_by_thresh:
                    self.scenario_averages[scenario][grid_size] = {
                        thresh: sum(accs) / len(accs) for thresh, accs in acc_by_thresh.items()
                    }
                
                # 计算平均匹配数
                if match_counts:
                    self.scenario_match_averages[scenario][grid_size] = sum(match_counts) / len(match_counts)

    
    def calculate_modality_averages(self):
        """计算每个模态在所有场景下的不同网格大小平均准确率"""
        for modality_pair in self.modality_pairs:
            for grid_size in self.grid_sizes:
                acc_by_thresh = defaultdict(list)
                match_counts = []

                for scenario in self.scenarios:
                    result_dict = self.scenario_modality_results.get(scenario, {}).get(modality_pair, {}).get(grid_size)
                    if result_dict and 'accuracy' in result_dict:
                        acc_dict = result_dict['accuracy']
                        for thresh, acc in acc_dict.items():
                            acc_by_thresh[thresh].append(acc)
                        
                        # 收集匹配数
                        match_counts.append(result_dict['match_count'])

                if acc_by_thresh:
                    self.modality_averages[modality_pair][grid_size] = {
                        thresh: sum(accs) / len(accs) for thresh, accs in acc_by_thresh.items()
                    }
                
                # 计算平均匹配数
                if match_counts:
                    self.modality_match_averages[modality_pair][grid_size] = sum(match_counts) / len(match_counts)

    
    def calculate_global_averages(self):
        """计算所有场景的不同网格平均准确率（每个像素阈值）"""
        for grid_size in self.grid_sizes:
            acc_by_thresh = defaultdict(list)
            match_counts = []

            for scenario in self.scenarios:
                for modality_pair in self.modality_pairs:
                    result_dict = self.scenario_modality_results.get(scenario, {}).get(modality_pair, {}).get(grid_size)
                    if result_dict and 'accuracy' in result_dict:
                        acc_dict = result_dict['accuracy']
                        for thresh, acc in acc_dict.items():
                            acc_by_thresh[thresh].append(acc)
                        
                        # 收集匹配数
                        match_counts.append(result_dict['match_count'])

            if acc_by_thresh:
                self.global_averages[grid_size] = {
                    thresh: sum(accs) / len(accs) for thresh, accs in acc_by_thresh.items()
                }
            
            # 计算全局平均匹配数
            if match_counts:
                self.global_match_averages[grid_size] = sum(match_counts) / len(match_counts)

    
    def run_analysis(self):
        """运行完整的分析流程"""
        print("开始分析匹配结果...")
        
        # 1. 扫描目录结构
        self.scan_directory_structure()
        
        # 2. 分析每个场景的每个模态配对
        print("\n正在分析各场景和模态配对的准确率...")
        for scenario in self.scenarios:
            print(f"分析场景 {scenario}:")
            for modality_pair in self.modality_pairs:
                self.analyze_scenario_modality(scenario, modality_pair)
        
        # 3. 计算各种平均值
        print("\n计算场景平均准确率...")
        self.calculate_scenario_averages()
        
        print("计算模态平均准确率...")
        self.calculate_modality_averages()
        
        print("计算全局平均准确率...")
        self.calculate_global_averages()
        
        print("\n分析完成！")
    
    def save_results_to_excel(self, output_file: str = "match_results_analysis.xlsx"):
        """将结果保存到Excel文件"""
        # 如果有排除模态，在文件名中体现
        if self.exclude_modalities:
            base_name, ext = os.path.splitext(output_file)
            excluded_str = "_excluded_" + "_".join(self.exclude_modalities)
            output_file = f"{base_name}{excluded_str}{ext}"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 1. 每个场景每个模态的详细结果（准确率）
            detailed_data = []
            for scenario in self.scenarios:
                for modality_pair in self.modality_pairs:
                    if scenario in self.scenario_modality_results and modality_pair in self.scenario_modality_results[scenario]:
                        row_data = {'场景': scenario, '模态配对': modality_pair}
                        for grid_size in self.grid_sizes:
                            if grid_size in self.scenario_modality_results[scenario][modality_pair]:
                                result_dict = self.scenario_modality_results[scenario][modality_pair][grid_size]
                                row_data[f'准确率_{grid_size}'] = result_dict.get('accuracy', None)
                            else:
                                row_data[f'准确率_{grid_size}'] = None
                        detailed_data.append(row_data)
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='详细结果_准确率', index=False)
            
            # 1.5. 每个场景每个模态的匹配数详细结果
            match_count_data = []
            for scenario in self.scenarios:
                for modality_pair in self.modality_pairs:
                    if scenario in self.scenario_match_counts and modality_pair in self.scenario_match_counts[scenario]:
                        row_data = {'场景': scenario, '模态配对': modality_pair}
                        for grid_size in self.grid_sizes:
                            if grid_size in self.scenario_match_counts[scenario][modality_pair]:
                                match_info = self.scenario_match_counts[scenario][modality_pair][grid_size]
                                row_data[f'有效匹配数_{grid_size}'] = match_info['match_count']
                                row_data[f'总匹配数_{grid_size}'] = match_info['total_matches']
                                row_data[f'保留率_{grid_size}'] = f"{match_info['retention_rate']:.2%}"
                            else:
                                row_data[f'有效匹配数_{grid_size}'] = None
                                row_data[f'总匹配数_{grid_size}'] = None
                                row_data[f'保留率_{grid_size}'] = None
                        match_count_data.append(row_data)
            
            match_count_df = pd.DataFrame(match_count_data)
            match_count_df.to_excel(writer, sheet_name='详细结果_匹配数', index=False)
            
            # 2. 场景平均准确率
            scenario_avg_data = []
            for scenario in self.scenarios:
                if scenario in self.scenario_averages:
                    row_data = {'场景': scenario}
                    for grid_size in self.grid_sizes:
                        if grid_size in self.scenario_averages[scenario]:
                            row_data[f'平均准确率_{grid_size}'] = self.scenario_averages[scenario][grid_size]
                        else:
                            row_data[f'平均准确率_{grid_size}'] = "NA"
                    scenario_avg_data.append(row_data)
            
            scenario_avg_df = pd.DataFrame(scenario_avg_data)
            scenario_avg_df.to_excel(writer, sheet_name='场景平均准确率', index=False)
            
            # 3. 模态平均准确率
            modality_avg_data = []
            for modality_pair in self.modality_pairs:
                if modality_pair in self.modality_averages:
                    row_data = {'模态配对': modality_pair}
                    for grid_size in self.grid_sizes:
                        if grid_size in self.modality_averages[modality_pair]:
                            row_data[f'平均准确率_{grid_size}'] = self.modality_averages[modality_pair][grid_size]
                        else:
                            row_data[f'平均准确率_{grid_size}'] = None
                    modality_avg_data.append(row_data)
            
            modality_avg_df = pd.DataFrame(modality_avg_data)
            modality_avg_df.to_excel(writer, sheet_name='模态平均准确率', index=False)
            
            # 4. 全局平均准确率
            global_avg_data = [{'网格大小': grid_size, '全局平均准确率': self.global_averages.get(grid_size, None)} 
                             for grid_size in self.grid_sizes]
            global_avg_df = pd.DataFrame(global_avg_data)
            global_avg_df.to_excel(writer, sheet_name='全局平均准确率', index=False)
            
            # 5. 场景平均匹配数
            scenario_match_avg_data = []
            for scenario in self.scenarios:
                if scenario in self.scenario_match_averages:
                    row_data = {'场景': scenario}
                    for grid_size in self.grid_sizes:
                        if grid_size in self.scenario_match_averages[scenario]:
                            row_data[f'平均匹配数_{grid_size}'] = self.scenario_match_averages[scenario][grid_size]
                        else:
                            row_data[f'平均匹配数_{grid_size}'] = "NA"
                    scenario_match_avg_data.append(row_data)
            
            scenario_match_avg_df = pd.DataFrame(scenario_match_avg_data)
            scenario_match_avg_df.to_excel(writer, sheet_name='场景平均匹配数', index=False)
            
            # 6. 模态平均匹配数
            modality_match_avg_data = []
            for modality_pair in self.modality_pairs:
                if modality_pair in self.modality_match_averages:
                    row_data = {'模态配对': modality_pair}
                    for grid_size in self.grid_sizes:
                        if grid_size in self.modality_match_averages[modality_pair]:
                            row_data[f'平均匹配数_{grid_size}'] = self.modality_match_averages[modality_pair][grid_size]
                        else:
                            row_data[f'平均匹配数_{grid_size}'] = None
                    modality_match_avg_data.append(row_data)
            
            modality_match_avg_df = pd.DataFrame(modality_match_avg_data)
            modality_match_avg_df.to_excel(writer, sheet_name='模态平均匹配数', index=False)
            
            # 7. 全局平均匹配数
            global_match_avg_data = [{'网格大小': grid_size, '全局平均匹配数': self.global_match_averages.get(grid_size, None)} 
                                   for grid_size in self.grid_sizes]
            global_match_avg_df = pd.DataFrame(global_match_avg_data)
            global_match_avg_df.to_excel(writer, sheet_name='全局平均匹配数', index=False)
        
        print(f"结果已保存到 {output_file}")
    
    def print_summary(self):
        """打印统计摘要"""
        print("\n" + "="*80)
        print("匹配结果统计摘要")
        print("="*80)

        grid_size_key = '8x8'

        # 全局平均准确率（8x8）
        print("\n【全局平均准确率 - 8x8】")
        if grid_size_key in self.global_averages:
            print(f"  8x8: {self.global_averages[grid_size_key]}")
        
        # 全局平均匹配数（8x8）
        print("\n【全局平均匹配数 - 8x8】")
        if grid_size_key in self.global_match_averages:
            print(f"  8x8: {self.global_match_averages[grid_size_key]:.1f}")

        # 获取第一个阈值（动态确定）
        first_threshold = None
        first_threshold_name = None
        for modality, grid_data in self.modality_averages.items():
            if grid_size_key in grid_data:
                acc_dict = grid_data[grid_size_key]
                if acc_dict:
                    first_threshold = min(acc_dict.keys())  # 取最小的阈值作为第一个阈值
                    first_threshold_name = f"{first_threshold}像素"
                    break
        
        if first_threshold is None:
            print("\n⚠️ 无法确定第一个阈值，跳过模态统计")
        else:
            # 模态第一阈值准确率（按8x8排序后前5个）
            print(f"\n【模态{first_threshold_name}阈值准确率 - 8x8（前5个）】")

            # 计算每个模态在8x8下的第一阈值准确率
            modality_first_8x8 = []
            for modality, grid_data in self.modality_averages.items():
                if grid_size_key in grid_data:
                    acc_dict = grid_data[grid_size_key]
                    if first_threshold in acc_dict:
                        acc_first = acc_dict[first_threshold]
                        modality_first_8x8.append((modality, acc_first))
        
            # 按第一阈值准确率排序，取前5
            top5_modalities = sorted(modality_first_8x8, key=lambda x: x[1], reverse=True)[:5]
            for modality, acc in top5_modalities:
                print(f"  {modality}: {first_threshold_name}阈值准确率 = {acc:.4f}")
                accs = self.modality_averages[modality][grid_size_key]
                for thresh, a in sorted(accs.items()):
                    print(f"    阈值 {thresh}px: {a:.4f}")
                
                # 显示平均匹配数
                if modality in self.modality_match_averages and grid_size_key in self.modality_match_averages[modality]:
                    avg_matches = self.modality_match_averages[modality][grid_size_key]
                    print(f"    平均匹配数: {avg_matches:.1f}")

        # 场景第一阈值准确率（使用与模态相同的第一阈值）
        if first_threshold is None:
            print("\n⚠️ 无法确定第一个阈值，跳过场景统计")
        else:
            print(f"\n【场景{first_threshold_name}阈值准确率 - 8x8（前5个）】")

            scenario_first_8x8 = []
            for scenario, grid_data in self.scenario_averages.items():
                if grid_size_key in grid_data:
                    acc_dict = grid_data[grid_size_key]
                    if first_threshold in acc_dict:
                        acc_first = acc_dict[first_threshold]
                        scenario_first_8x8.append((scenario, acc_first))

            # 按第一阈值准确率排序，取前5
            top5_scenarios = sorted(scenario_first_8x8, key=lambda x: x[1], reverse=True)[:5]
            for scenario, acc in top5_scenarios:
                print(f"  {scenario}: {first_threshold_name}阈值准确率 = {acc:.4f}")
                accs = self.scenario_averages[scenario][grid_size_key]
                for thresh, a in sorted(accs.items()):
                    print(f"    阈值 {thresh}px: {a:.4f}")
                
                # 显示平均匹配数
                if scenario in self.scenario_match_averages and grid_size_key in self.scenario_match_averages[scenario]:
                    avg_matches = self.scenario_match_averages[scenario][grid_size_key]
                    print(f"    平均匹配数: {avg_matches:.1f}")

        print("\n详细结果请查看生成的Excel文件。")
        print("新增功能：现在包含RANSAC算法剔除后的匹配数统计！")

def main():
    """主函数"""
    import argparse
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='多模态匹配结果准确率统计脚本')
    parser.add_argument('--output_dir', 
                       default="/media/jackew/Extreme SSD/newData/MultiModal_Output",
                       help='输出目录路径 (默认: /media/jackew/Extreme SSD/newData/MultiModal_Output)')
    parser.add_argument('--exclude_modalities', 
                       nargs='*', 
                       default=[],
                       help='要排除的模态列表，例如: --exclude_modalities OpticalFlow FreqDomain')
    parser.add_argument('--excel_output',
                       default="match_results_analysis.xlsx",
                       help='Excel输出文件名 (默认: match_results_analysis.xlsx)')
    parser.add_argument('--scene_prefix',
                       default='V',
                       help='场景文件夹前缀，例如 V, A, K 等 (默认: V)')
    parser.add_argument('--use_perspective_label',
                       action='store_true',
                       help='使用透视变换标签矩阵计算准确性（用于透视变换增强的数据）')
    
    args = parser.parse_args()
    
    # 设置输出目录路径
    output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        print(f"错误：输出目录不存在 - {output_dir}")
        return
    
    # 打印配置信息
    print("="*80)
    print("多模态匹配结果分析配置")
    print("="*80)
    print(f"输出目录: {output_dir}")
    print(f"排除模态: {args.exclude_modalities if args.exclude_modalities else '无'}")
    print(f"Excel输出: {args.excel_output}")
    print(f"场景前缀: {args.scene_prefix}")
    print(f"透视变换标签: {'启用' if args.use_perspective_label else '禁用'}")
    if args.use_perspective_label:
        print("  ⚠️ 注意：启用透视变换标签需要相对变换矩阵文件 (relative_transform_matrix.npy)")
    print("="*80)
    
    # 创建分析器并运行分析
    analyzer = MatchResultAnalyzer(output_dir, exclude_modalities=args.exclude_modalities, 
                                 scene_prefix=args.scene_prefix, use_perspective_label=args.use_perspective_label)
    analyzer.run_analysis()
    
    # 保存结果并打印摘要
    analyzer.save_results_to_excel(args.excel_output)
    analyzer.print_summary()

if __name__ == "__main__":
    main()
