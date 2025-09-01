#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态匹配结果准确率统计脚本
分析 MultiModal_Output 目录下的匹配结果

使用示例:
1. 分析所有模态:
   python analyze_match_results.py

2. 排除OpticalFlow模态:
   python analyze_match_results.py --exclude_modalities OpticalFlow

3. 排除多个模态:
   python analyze_match_results.py --exclude_modalities OpticalFlow FreqDomain

4. 指定自定义输出目录和Excel文件名:
   python analyze_match_results.py --output_dir /path/to/output --excel_output custom_results.xlsx

5. 查看帮助信息:
   python analyze_match_results.py --help
"""

import os
import glob
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import json
import cv2


def get_video_resolution_from_match_path(match_path: str, video_root: str = "/media/jackew/Extreme SSD/newData/resized") -> tuple:
    """
    从匹配结果路径中提取场景名，读取对应视频的分辨率。
    
    Args:
        match_path: 匹配结果文件路径，例如 MultiModal_Output_ransac_5_05/K1/...
        video_root: 视频根目录，默认 /media/jackew/Extreme SSD/oldData/renamed
    
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


def calculate_accuracy(match_file_path: str) -> Dict[int, float]:

    use_pt_level = True
    """
    使用单位矩阵评估匹配准确率，误差阈值从 1 到 10 像素。
    不使用 RANSAC，仅评估变换误差。

    Returns:
        Dict[int, float]: key 是像素误差阈值（1-10），value 是该阈值下的准确率
    """
    try:
        matches = load_match_ids_from_txt(match_file_path)
        if not matches or len(matches) < 1:
            print(f"❌ 匹配数据为空或无效: {matches}")
            return {k: 0.0 for k in range(1, 11)}

        filename = os.path.basename(match_file_path)
        if 'x' not in filename:
            print(f"❌ 文件名无法识别 patch size: {filename}")
            return {k: 0.0 for k in range(1, 11)}

        patch_size = int(filename.split('x')[0])
        res = get_video_resolution_from_match_path(match_file_path)
        if res is None:
            return {k: 0.0 for k in range(1, 11)}
        width, height = res

        # 计算真实坐标
        src_pts = np.array([patch_id_to_coords(m[0], patch_size, width) for m in matches])
        dst_pts = np.array([patch_id_to_coords(m[1], patch_size, width) for m in matches])

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

        if mask is None:
            print(f"⚠️ RANSAC 返回无效 mask: {match_file_path}")
            return {k: 0.0 for k in range(1, 11)}

        # 提取内点
        inlier_src = src_pts[mask.ravel() == 1]
        inlier_dst = dst_pts[mask.ravel() == 1]
        

        # # 使用单位矩阵，即不进行任何变换
        # transformed_pts = src_pts  # 因为 H = I

        # 计算欧几里得误差
        if not use_pt_level:
            errors = np.linalg.norm(inlier_src - inlier_dst, axis=1)
            total = len(errors)
            acc_by_thresh = {}
            for thresh in range(1, 11):
                inliers = np.sum(errors <= thresh)
                acc_by_thresh[thresh] = inliers / total if total > 0 else 0.0

            return acc_by_thresh
        else:
            dx = np.abs(inlier_src[:, 0] - inlier_dst[:, 0])
            dy = np.abs(inlier_src[:, 1] - inlier_dst[:, 1])

            errors = np.maximum(dx, dy)
            total = len(errors)
            acc_by_thresh = {}
            for thresh in [4,8,12,16,20]:
                inliers = np.sum(errors <= thresh)
                acc_by_thresh[thresh] = inliers / total if total > 0 else 0.0

            return acc_by_thresh

    except Exception as e:
        print(f"❌ 处理失败: {match_file_path} - {str(e)}")
        return {k: 0.0 for k in range(1, 11)}




# def calculate_accuracy(match_file_path: str) -> float:
#     """
#     计算匹配结果文件的准确率
#     这个函数需要根据实际的准确率计算方法来实现
#     现在作为占位符，返回一个示例值
    
#     Args:
#         match_file_path: 匹配结果文件路径
        
#     Returns:
#         准确率值 (0-1之间的浮点数)
#     """
#     # TODO: 实现具体的准确率计算逻辑
#     # 这里只是一个占位符，需要根据实际的匹配结果文件格式来实现
#     try:
#         with open(match_file_path, 'r') as f:
#             lines = f.readlines()
#             # 这里需要实现具体的准确率计算逻辑
#             # 现在返回一个基于文件行数的示例值
#             return min(1.0, len(lines) / 100.0)
#     except Exception as e:
#         print(f"Error reading {match_file_path}: {e}")
#         return 0.0

class MatchResultAnalyzer:
    def __init__(self, output_dir: str, exclude_modalities: List[str] = None):
        """
        初始化分析器
        
        Args:
            output_dir: MultiModal_Output_ransac_5_05 目录路径
            exclude_modalities: 要排除的模态列表，例如 ['OpticalFlow', 'FreqDomain']
        """
        self.output_dir = output_dir
        self.exclude_modalities = exclude_modalities or []
        self.scenarios = []
        self.modality_pairs = []
        self.grid_sizes = ['8x8', '16x16', '32x32', '64x64']
        
        # 存储所有统计结果
        self.scenario_modality_results = defaultdict(lambda: defaultdict(dict))  # scenario -> modality_pair -> grid_size -> accuracy
        self.scenario_averages = defaultdict(dict)  # scenario -> grid_size -> average_accuracy
        self.modality_averages = defaultdict(dict)  # modality_pair -> grid_size -> average_accuracy
        self.global_averages = {}  # grid_size -> average_accuracy
        
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
            if os.path.isdir(os.path.join(self.output_dir, item)) and item.startswith('V'):
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
            acc_dict = calculate_accuracy(file_path)
            self.scenario_modality_results[scenario][modality_pair][grid_size] = acc_dict
            print(f"  {scenario} - {modality_pair} - {grid_size}: {acc_dict}")
    
    def calculate_scenario_averages(self):
        """计算每个场景所有模态匹配结果的不同网格大小平均准确率"""
        for scenario in self.scenarios:
            for grid_size in self.grid_sizes:
                acc_by_thresh = defaultdict(list)  # 每个像素阈值 -> 所有模态准确率列表

                for modality_pair in self.modality_pairs:
                    acc_dict = self.scenario_modality_results.get(scenario, {}).get(modality_pair, {}).get(grid_size)
                    if acc_dict:
                        for thresh, acc in acc_dict.items():
                            acc_by_thresh[thresh].append(acc)

                # 计算每个像素阈值的平均准确率
                if acc_by_thresh:
                    self.scenario_averages[scenario][grid_size] = {
                        thresh: sum(accs) / len(accs) for thresh, accs in acc_by_thresh.items()
                    }

    
    def calculate_modality_averages(self):
        """计算每个模态在所有场景下的不同网格大小平均准确率"""
        for modality_pair in self.modality_pairs:
            for grid_size in self.grid_sizes:
                acc_by_thresh = defaultdict(list)

                for scenario in self.scenarios:
                    acc_dict = self.scenario_modality_results.get(scenario, {}).get(modality_pair, {}).get(grid_size)
                    if acc_dict:
                        for thresh, acc in acc_dict.items():
                            acc_by_thresh[thresh].append(acc)

                if acc_by_thresh:
                    self.modality_averages[modality_pair][grid_size] = {
                        thresh: sum(accs) / len(accs) for thresh, accs in acc_by_thresh.items()
                    }

    
    def calculate_global_averages(self):
        """计算所有场景的不同网格平均准确率（每个像素阈值）"""
        for grid_size in self.grid_sizes:
            acc_by_thresh = defaultdict(list)

            for scenario in self.scenarios:
                for modality_pair in self.modality_pairs:
                    acc_dict = self.scenario_modality_results.get(scenario, {}).get(modality_pair, {}).get(grid_size)
                    if acc_dict:
                        for thresh, acc in acc_dict.items():
                            acc_by_thresh[thresh].append(acc)

            if acc_by_thresh:
                self.global_averages[grid_size] = {
                    thresh: sum(accs) / len(accs) for thresh, accs in acc_by_thresh.items()
                }

    
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
            # 1. 每个场景每个模态的详细结果
            detailed_data = []
            for scenario in self.scenarios:
                for modality_pair in self.modality_pairs:
                    if scenario in self.scenario_modality_results and modality_pair in self.scenario_modality_results[scenario]:
                        row_data = {'场景': scenario, '模态配对': modality_pair}
                        for grid_size in self.grid_sizes:
                            if grid_size in self.scenario_modality_results[scenario][modality_pair]:
                                row_data[f'准确率_{grid_size}'] = self.scenario_modality_results[scenario][modality_pair][grid_size]
                            else:
                                row_data[f'准确率_{grid_size}'] = None
                        detailed_data.append(row_data)
            
            detailed_df = pd.DataFrame(detailed_data)
            detailed_df.to_excel(writer, sheet_name='详细结果', index=False)
            
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

        # 模态平均准确率（按8x8排序后前5个）
        print("\n【模态平均准确率 - 8x8（前5个）】")

        # 计算每个模态在8x8下的平均准确率（平均所有阈值）
        modality_avg_8x8 = []
        for modality, grid_data in self.modality_averages.items():
            if grid_size_key in grid_data:
                acc_dict = grid_data[grid_size_key]
                mean_acc = np.mean(list(acc_dict.values()))
                modality_avg_8x8.append((modality, mean_acc))
    
        # 按平均准确率排序，取前5
        top5_modalities = sorted(modality_avg_8x8, key=lambda x: x[1], reverse=True)[:5]
        for modality, acc in top5_modalities:
            print(f"  {modality}: 平均准确率 = {acc:.4f}")
            accs = self.modality_averages[modality][grid_size_key]
            for thresh, a in sorted(accs.items()):
                print(f"    阈值 {thresh}px: {a:.4f}")

        # 场景平均准确率（按8x8排序后前5个）
        print("\n【场景平均准确率 - 8x8（前5个）】")

        scenario_avg_8x8 = []
        for scenario, grid_data in self.scenario_averages.items():
            if grid_size_key in grid_data:
                acc_dict = grid_data[grid_size_key]
                mean_acc = np.mean(list(acc_dict.values()))
                scenario_avg_8x8.append((scenario, mean_acc))

        # 排序 + 取前5
        top5_scenarios = sorted(scenario_avg_8x8, key=lambda x: x[1], reverse=True)[:5]
        for scenario, acc in top5_scenarios:
            print(f"  {scenario}: 平均准确率 = {acc:.4f}")
            accs = self.scenario_averages[scenario][grid_size_key]
            for thresh, a in sorted(accs.items()):
                print(f"    阈值 {thresh}px: {a:.4f}")

        print("\n详细结果请查看生成的Excel文件。")

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
    print("="*80)
    
    # 创建分析器并运行分析
    analyzer = MatchResultAnalyzer(output_dir, exclude_modalities=args.exclude_modalities)
    analyzer.run_analysis()
    
    # 保存结果并打印摘要
    analyzer.save_results_to_excel(args.excel_output)
    analyzer.print_summary()

if __name__ == "__main__":
    main()
