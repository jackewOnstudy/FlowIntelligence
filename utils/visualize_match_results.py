#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态匹配结果可视化脚本 - 基于真实数据
实现方案3：分层展示 - 热力图 + 关键切面折线图

特点:
- 直接从原始匹配文件计算真实的按百分比准确率
- 所有图表数据都基于真实的RANSAC计算结果
- 支持多场景、多模态配对的数据聚合展示

依赖安装:
pip install matplotlib seaborn plotly kaleido pandas numpy opencv-python

使用方法:
1. 先运行 analyze_match_results.py 生成Excel分析结果
2. 运行此脚本生成可视化图表:
   python visualize_match_results.py --data_dir /path/to/original/data --excel_file match_results_analysis.xlsx
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
from typing import Dict, List, Tuple, Optional
import sys
sys.path.append('.')
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和美观主题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class MatchResultVisualizer:
    def __init__(self, excel_file: str, data_dir: str, scene_prefix: str = 'V'):
        """
        初始化可视化器
        
        Args:
            excel_file: Excel分析结果文件路径（用于获取模态配对信息）
            data_dir: 原始数据目录路径（必需，用于计算真实的百分比准确率）
            scene_prefix: 场景文件夹前缀，例如 'V', 'A', 'K' 等
        """
        self.excel_file = excel_file
        self.data_dir = data_dir
        self.scene_prefix = scene_prefix
        self.data = {}
        self.grid_sizes = ['8x8', '16x16', '32x32', '64x64']
        self.thresholds = [4, 8, 12, 16, 20]  # 像素阈值
        self.match_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 匹配百分比
        
        # 专业配色方案
        self.colors = {
            'primary': '#2E86AB',      # 深蓝
            'secondary': '#A23B72',    # 紫红
            'accent': '#F18F01',       # 橙色
            'success': '#C73E1D',      # 红色
            'info': '#6A994E',         # 绿色
            'background': '#F5F7FA',   # 浅灰背景
            'text': '#2D3748'          # 深灰文字
        }
        
    def load_data(self):
        """从Excel文件加载数据"""
        try:
            # 读取各个工作表
            self.data['detailed_accuracy'] = pd.read_excel(self.excel_file, sheet_name='详细结果_准确率')
            self.data['detailed_matches'] = pd.read_excel(self.excel_file, sheet_name='详细结果_匹配数')
            self.data['modality_avg_accuracy'] = pd.read_excel(self.excel_file, sheet_name='模态平均准确率')
            self.data['scenario_avg_accuracy'] = pd.read_excel(self.excel_file, sheet_name='场景平均准确率')
            self.data['global_avg_accuracy'] = pd.read_excel(self.excel_file, sheet_name='全局平均准确率')
            
            print("✅ 数据加载成功")
            print(f"场景数量: {len(self.data['detailed_accuracy']['场景'].unique())}")
            print(f"模态配对数量: {len(self.data['detailed_accuracy']['模态配对'].unique())}")
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise
    
    def parse_accuracy_dict(self, acc_str: str) -> Dict[int, float]:
        """解析准确率字典字符串"""
        if pd.isna(acc_str) or acc_str == "NA":
            return {}
        try:
            # 假设格式类似 "{4: 0.85, 8: 0.92, 12: 0.94, 16: 0.96, 20: 0.97}"
            acc_dict = eval(acc_str) if isinstance(acc_str, str) else acc_str
            return acc_dict if isinstance(acc_dict, dict) else {}
        except:
            return {}
    


    def calculate_real_percentage_data(self, match_file_path: str) -> Dict[int, Dict[int, float]]:
        """
        从原始匹配文件计算真实的按百分比准确率数据
        
        Args:
            match_file_path: 匹配结果文件路径
            
        Returns:
            Dict[percentage, Dict[threshold, accuracy]]
        """
        try:
            # 重用 analyze_match_results.py 中的函数
            from analyze_match_results import load_match_ids_from_txt, patch_id_to_coords, get_video_resolution_from_match_path
            import cv2
            
            # 读取匹配数据
            matches = load_match_ids_from_txt(match_file_path)
            if not matches:
                return {}
            
            # 获取视频分辨率和patch大小
            filename = os.path.basename(match_file_path)
            if 'x' not in filename:
                return {}
            
            patch_size = int(filename.split('x')[0])
            res = get_video_resolution_from_match_path(match_file_path)
            if res is None:
                return {}
            
            width, height = res
            
            # 计算坐标
            src_pts = np.array([patch_id_to_coords(m[0], patch_size, width) for m in matches])
            dst_pts = np.array([patch_id_to_coords(m[1], patch_size, width) for m in matches])
            
            result_data = {}
            
            # 对每个百分比计算准确率
            for pct in self.match_percentages:
                # 取前pct%的匹配
                num_matches = int(len(matches) * pct / 100)
                if num_matches < 4:  # RANSAC最少需要4个点
                    continue
                
                subset_src = src_pts[:num_matches]
                subset_dst = dst_pts[:num_matches]
                
                # 检查匹配点数量是否足够进行RANSAC
                if num_matches < 4:
                    # 匹配点数量不足，跳过RANSAC，直接计算准确率
                    inlier_src = subset_src
                    inlier_dst = subset_dst
                else:
                    # 使用RANSAC计算内点
                    H, mask = cv2.findHomography(
                        subset_src, subset_dst,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=max(3.0, patch_size * 0.5),
                        maxIters=2000,
                        confidence=0.95
                    )
                    
                    if mask is None:
                        # RANSAC失败，使用所有匹配点
                        inlier_src = subset_src
                        inlier_dst = subset_dst
                    else:
                        # 提取内点
                        inlier_mask = mask.ravel() == 1
                        inlier_src = subset_src[inlier_mask]
                        inlier_dst = subset_dst[inlier_mask]
                        
                        if len(inlier_src) == 0:
                            # 没有内点，使用所有匹配点
                            inlier_src = subset_src
                            inlier_dst = subset_dst
                
                # 计算不同阈值下的准确率
                threshold_acc = {}
                for threshold in self.thresholds:
                    # 计算像素级误差 (使用 max(dx, dy) 与 analyze_match_results.py 保持一致)
                    dx = np.abs(inlier_src[:, 0] - inlier_dst[:, 0])
                    dy = np.abs(inlier_src[:, 1] - inlier_dst[:, 1])
                    errors = np.maximum(dx, dy)
                    
                    # 计算准确率
                    inliers = np.sum(errors <= threshold)
                    accuracy = inliers / len(errors) if len(errors) > 0 else 0.0
                    threshold_acc[threshold] = accuracy
                
                # 检查是否所有阈值下的准确率都为0，如果是则重新计算
                if sum(threshold_acc.values()) == 0:
                    # 正确匹配过少，RANSAC筛选出来的inlier point其实是错误的，使用所有原始匹配点
                    inlier_src = subset_src
                    inlier_dst = subset_dst
                    
                    # 重新计算准确率
                    threshold_acc = {}
                    for threshold in self.thresholds:
                        dx = np.abs(inlier_src[:, 0] - inlier_dst[:, 0])
                        dy = np.abs(inlier_src[:, 1] - inlier_dst[:, 1])
                        errors = np.maximum(dx, dy)
                        
                        inliers = np.sum(errors <= threshold)
                        accuracy = inliers / len(errors) if len(errors) > 0 else 0.0
                        threshold_acc[threshold] = accuracy
                
                result_data[pct] = threshold_acc
            
            return result_data
            
        except Exception as e:
            print(f"⚠️ 无法计算真实百分比数据: {e}")
            return {}
    
    def create_heatmap_data(self) -> pd.DataFrame:
        """创建热力图数据 - 使用真实数据"""
        heatmap_data = []
        
        print("🔍 从原始匹配文件计算真实的百分比准确率...")
        
        # 获取所有场景和模态配对
        scenarios = []
        modality_pairs = []
        
        # 从Excel文件中获取场景和模态配对信息
        if 'detailed_accuracy' in self.data:
            scenarios = self.data['detailed_accuracy']['场景'].unique().tolist()
            modality_pairs = self.data['detailed_accuracy']['模态配对'].unique().tolist()
        else:
            # 如果Excel数据不可用，直接扫描数据目录
            scenarios, modality_pairs = self._scan_data_directory()
        
        # 对每个场景、模态配对和网格大小计算真实准确率
        total_combinations = len(scenarios) * len(modality_pairs) * len(self.grid_sizes)
        current_count = 0
        
        for scenario in scenarios:
            for modality_pair in modality_pairs:
                for grid_size in self.grid_sizes:
                    current_count += 1
                    print(f"处理进度: {current_count}/{total_combinations} - {scenario}/{modality_pair}/{grid_size}")
                    
                    # 找到对应的匹配文件
                    match_file_path = self._find_match_file(scenario, modality_pair, grid_size)
                    
                    if match_file_path and os.path.exists(match_file_path):
                        # 计算真实的百分比准确率数据
                        pct_accuracy_data = self.calculate_real_percentage_data(match_file_path)
                        
                        # 将数据添加到热力图数据中
                        for pct in self.match_percentages:
                            if pct in pct_accuracy_data:
                                for threshold in self.thresholds:
                                    if threshold in pct_accuracy_data[pct]:
                                        accuracy = pct_accuracy_data[pct][threshold]
                                        heatmap_data.append({
                                            'Scenario': scenario,
                                            'Modality Pair': modality_pair,
                                            'Grid Size': grid_size,
                                            'Error Threshold (px)': threshold,
                                            'Match Percentage (%)': pct,
                                            'Accuracy': accuracy
                                        })
        
        return pd.DataFrame(heatmap_data)
    
    def _scan_data_directory(self) -> Tuple[List[str], List[str]]:
        """扫描数据目录获取场景和模态配对"""
        scenarios = []
        modality_pairs = set()
        
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path) and item.startswith(self.scene_prefix):
                scenarios.append(item)
                
                # 扫描模态配对
                for modality_item in os.listdir(item_path):
                    modality_path = os.path.join(item_path, modality_item)
                    if os.path.isdir(modality_path) and '_vs_' in modality_item:
                        modality_pairs.add(modality_item)
        
        scenarios.sort()
        modality_pairs = sorted(list(modality_pairs))
        
        print(f"📊 发现 {len(scenarios)} 个场景: {scenarios[:5]}{'...' if len(scenarios) > 5 else ''}")
        print(f"📊 发现 {len(modality_pairs)} 个模态配对: {modality_pairs[:3]}{'...' if len(modality_pairs) > 3 else ''}")
        
        return scenarios, modality_pairs
    
    def _find_match_file(self, scenario: str, modality_pair: str, grid_size: str) -> Optional[str]:
        """找到对应的匹配文件路径"""
        # 构建文件路径: data_dir/scenario/modality_pair/MatchResult/List/subdir/grid_size.txt
        base_path = os.path.join(self.data_dir, scenario, modality_pair, 'MatchResult', 'List')
        
        if not os.path.exists(base_path):
            return None
        
        try:
            # 找到List下的子文件夹
            subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            if not subdirs:
                return None
            
            # 取第一个子文件夹
            subdir = subdirs[0]
            match_file = os.path.join(base_path, subdir, f"{grid_size}.txt")
            
            return match_file if os.path.exists(match_file) else None
            
        except Exception as e:
            print(f"⚠️ 扫描路径出错 {base_path}: {e}")
            return None
    
    def plot_main_heatmap(self, data: pd.DataFrame, output_dir: str):
        """绘制主热力图"""
        # 为每个网格大小创建热力图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multimodal Matching Accuracy Heatmap\n(Error Threshold vs Match Percentage)', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, grid_size in enumerate(self.grid_sizes):
            grid_data = data[data['Grid Size'] == grid_size]
            
            if grid_data.empty:
                continue
            
            # 对多个场景和模态配对的数据进行平均
            aggregated_data = grid_data.groupby(['Error Threshold (px)', 'Match Percentage (%)'])['Accuracy'].mean().reset_index()
                
            # 创建透视表
            pivot_data = aggregated_data.pivot(index='Error Threshold (px)', 
                                             columns='Match Percentage (%)', 
                                             values='Accuracy')
            
            # 绘制热力图
            im = sns.heatmap(pivot_data, 
                           annot=True, 
                           fmt='.3f',
                           cmap='RdYlBu_r',
                           ax=axes[idx],
                           cbar_kws={'label': 'Accuracy'},
                           annot_kws={'size': 8})
            
            axes[idx].set_title(f'Grid Size: {grid_size}', fontweight='bold', fontsize=12)
            axes[idx].set_xlabel('Match Percentage (%)', fontweight='bold')
            axes[idx].set_ylabel('Error Threshold (pixels)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmap_accuracy_overview.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ 主热力图已生成: heatmap_accuracy_overview.png")
    
    def plot_threshold_curves(self, data: pd.DataFrame, output_dir: str):
        """绘制关键阈值曲线图"""
        key_thresholds = [4, 8, 16]  # 选择代表性阈值
        key_grid = '8x8'  # 重点关注8x8网格
        
        plt.figure(figsize=(12, 8))
        
        grid_data = data[data['Grid Size'] == key_grid]
        
        # 对数据进行聚合
        aggregated_data = grid_data.groupby(['Error Threshold (px)', 'Match Percentage (%)'])['Accuracy'].mean().reset_index()
        
        for threshold in key_thresholds:
            thresh_data = aggregated_data[aggregated_data['Error Threshold (px)'] == threshold]
            if not thresh_data.empty:
                plt.plot(thresh_data['Match Percentage (%)'], 
                        thresh_data['Accuracy'],
                        marker='o', 
                        linewidth=2.5,
                        markersize=6,
                        label=f'{threshold}px Threshold',
                        alpha=0.8)
        
        plt.title(f'Accuracy vs Match Percentage (Grid: {key_grid})', fontweight='bold', fontsize=14)
        plt.xlabel('Match Percentage (%)', fontweight='bold', fontsize=12)
        plt.ylabel('Accuracy', fontweight='bold', fontsize=12)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # 添加最佳工作点标注
        for threshold in key_thresholds:
            thresh_data = aggregated_data[aggregated_data['Error Threshold (px)'] == threshold]
            if not thresh_data.empty:
                # 找到准确率最高的点
                best_idx = thresh_data['Accuracy'].idxmax()
                best_row = thresh_data.loc[best_idx]
                plt.annotate(f'Best Point\n({best_row["Match Percentage (%)"]}%, {best_row["Accuracy"]:.3f})',
                           xy=(best_row['Match Percentage (%)'], best_row['Accuracy']),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'threshold_curves.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ 阈值曲线图已生成: threshold_curves.png")
    
    def plot_percentage_curves(self, data: pd.DataFrame, output_dir: str):
        """绘制关键百分比曲线图"""
        key_percentages = [25, 50, 75]  # 选择代表性百分比
        key_grid = '8x8'
        
        plt.figure(figsize=(12, 8))
        
        grid_data = data[data['Grid Size'] == key_grid]
        
        # 对数据进行聚合
        aggregated_data = grid_data.groupby(['Error Threshold (px)', 'Match Percentage (%)'])['Accuracy'].mean().reset_index()
        
        for pct in key_percentages:
            pct_data = aggregated_data[aggregated_data['Match Percentage (%)'] == pct]
            if not pct_data.empty:
                plt.plot(pct_data['Error Threshold (px)'], 
                        pct_data['Accuracy'],
                        marker='s', 
                        linewidth=2.5,
                        markersize=6,
                        label=f'Top {pct}% Matches',
                        alpha=0.8)
        
        plt.title(f'Accuracy vs Error Threshold (Grid: {key_grid})', fontweight='bold', fontsize=14)
        plt.xlabel('Error Threshold (pixels)', fontweight='bold', fontsize=12)
        plt.ylabel('Accuracy', fontweight='bold', fontsize=12)
        plt.legend(frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'percentage_curves.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("✅ 百分比曲线图已生成: percentage_curves.png")
    
    def plot_interactive_3d(self, data: pd.DataFrame, output_dir: str):
        """创建交互式3D图表"""
        key_grid = '8x8'
        grid_data = data[data['Grid Size'] == key_grid]
        
        if grid_data.empty:
            return
        
        # 对数据进行聚合
        aggregated_data = grid_data.groupby(['Error Threshold (px)', 'Match Percentage (%)'])['Accuracy'].mean().reset_index()
        
        # 创建3D散点图
        fig = go.Figure(data=[go.Scatter3d(
            x=aggregated_data['Match Percentage (%)'],
            y=aggregated_data['Error Threshold (px)'],
            z=aggregated_data['Accuracy'],
            mode='markers',
            marker=dict(
                size=8,
                color=aggregated_data['Accuracy'],
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Accuracy")
            ),
            text=[f'Percentage: {p}%<br>Threshold: {t}px<br>Accuracy: {a:.3f}' 
                  for p, t, a in zip(aggregated_data['Match Percentage (%)'], 
                                    aggregated_data['Error Threshold (px)'], 
                                    aggregated_data['Accuracy'])],
            hovertemplate='%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title=f'3D Match Result Analysis (Grid: {key_grid})',
            scene=dict(
                xaxis_title='Match Percentage (%)',
                yaxis_title='Error Threshold (pixels)',
                zaxis_title='Accuracy',
                bgcolor='white'
            ),
            width=800,
            height=600
        )
        
        # 保存为HTML
        fig.write_html(os.path.join(output_dir, 'interactive_3d_plot.html'))
        print("✅ 交互式3D图已生成: interactive_3d_plot.html")
    
    def plot_modality_comparison(self, output_dir: str):
        """绘制模态对比图"""
        plt.figure(figsize=(14, 10))
        
        # 解析模态平均准确率数据
        modality_data = []
        for _, row in self.data['modality_avg_accuracy'].iterrows():
            modality = row['模态配对']
            for grid_size in self.grid_sizes:
                acc_col = f'平均准确率_{grid_size}'
                if acc_col in row and not pd.isna(row[acc_col]):
                    acc_dict = self.parse_accuracy_dict(row[acc_col])
                    if acc_dict:
                        # 计算平均准确率
                        avg_acc = np.mean(list(acc_dict.values()))
                        modality_data.append({
                            '模态配对': modality,
                            '网格大小': grid_size,
                            '平均准确率': avg_acc
                        })
        
        if modality_data:
            df_modality = pd.DataFrame(modality_data)
            
            # 选择前10个模态进行可视化
            top_modalities = df_modality.groupby('模态配对')['平均准确率'].mean().nlargest(10).index
            df_plot = df_modality[df_modality['模态配对'].isin(top_modalities)]
            
            # 创建分组柱状图
            pivot_modality = df_plot.pivot(index='模态配对', columns='网格大小', values='平均准确率')
            
            ax = pivot_modality.plot(kind='bar', 
                                   figsize=(14, 8), 
                                   color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'],
                                   alpha=0.8)
            
            plt.title('Accuracy Comparison of Different Modality Pairs (Top 10)', fontweight='bold', fontsize=14)
            plt.xlabel('Modality Pairs', fontweight='bold', fontsize=12)
            plt.ylabel('Average Accuracy', fontweight='bold', fontsize=12)
            plt.legend(title='Grid Size', frameon=True, fancybox=True, shadow=True)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'modality_comparison.png'), 
                       dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print("✅ 模态对比图已生成: modality_comparison.png")
    
    def generate_summary_dashboard(self, data: pd.DataFrame, output_dir: str):
        """生成综合仪表板"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Global Accuracy Heatmap', '8x8 Grid Threshold Curves', '8x8 Grid Percentage Impact', 'Grid Size Comparison'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. 热力图 (使用8x8数据)
        key_grid = '8x8'
        grid_data = data[data['Grid Size'] == key_grid]
        if not grid_data.empty:
            # 对数据进行聚合
            aggregated_data = grid_data.groupby(['Error Threshold (px)', 'Match Percentage (%)'])['Accuracy'].mean().reset_index()
            pivot_data = aggregated_data.pivot(index='Error Threshold (px)', 
                                             columns='Match Percentage (%)', 
                                             values='Accuracy')
            
            fig.add_trace(
                go.Heatmap(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    colorscale='RdYlBu_r',
                    showscale=False
                ),
                row=1, col=1
            )
        
        # 2. 阈值曲线
        for threshold in [4, 8, 16]:
            thresh_data = aggregated_data[aggregated_data['Error Threshold (px)'] == threshold]
            if not thresh_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=thresh_data['Match Percentage (%)'],
                        y=thresh_data['Accuracy'],
                        mode='lines+markers',
                        name=f'{threshold}px',
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # 3. 百分比影响
        for pct in [25, 50, 75]:
            pct_data = aggregated_data[aggregated_data['Match Percentage (%)'] == pct]
            if not pct_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pct_data['Error Threshold (px)'],
                        y=pct_data['Accuracy'],
                        mode='lines+markers',
                        name=f'{pct}%',
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # 4. 网格大小对比
        grid_comparison = []
        for grid_size in self.grid_sizes:
            grid_subset = data[data['Grid Size'] == grid_size]
            if not grid_subset.empty:
                avg_acc = grid_subset['Accuracy'].mean()
                grid_comparison.append((grid_size, avg_acc))
        
        if grid_comparison:
            grid_names, grid_accs = zip(*grid_comparison)
            fig.add_trace(
                go.Bar(
                    x=list(grid_names),
                    y=list(grid_accs),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Multimodal Matching Results - Comprehensive Analysis Dashboard",
            title_x=0.5
        )
        
        fig.write_html(os.path.join(output_dir, 'dashboard.html'))
        print("✅ 综合仪表板已生成: dashboard.html")
    
    def run_visualization(self, output_dir: str = "visualization_output"):
        """运行完整的可视化流程"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print("🎨 开始生成可视化图表...")
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 创建热力图数据
        heatmap_data = self.create_heatmap_data()
        
        # 3. 生成各种图表
        self.plot_main_heatmap(heatmap_data, output_dir)
        self.plot_threshold_curves(heatmap_data, output_dir)
        self.plot_percentage_curves(heatmap_data, output_dir)
        self.plot_interactive_3d(heatmap_data, output_dir)
        self.plot_modality_comparison(output_dir)
        self.generate_summary_dashboard(heatmap_data, output_dir)
        
        print(f"\n🎉 所有可视化图表已生成完成！")
        print(f"📁 输出目录: {output_dir}")
        print("📊 生成的文件:")
        print("  - heatmap_accuracy_overview.png: 主热力图")
        print("  - threshold_curves.png: 阈值曲线图")
        print("  - percentage_curves.png: 百分比曲线图")
        print("  - modality_comparison.png: 模态对比图")
        print("  - interactive_3d_plot.html: 交互式3D图")
        print("  - dashboard.html: 综合仪表板")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='多模态匹配结果可视化脚本 - 使用真实数据')
    parser.add_argument('--excel_file', 
                       default="match_results_analysis.xlsx",
                       help='Excel分析结果文件路径（用于获取模态配对信息）')
    parser.add_argument('--output_dir',
                       default="./results/visualization_output",
                       help='输出目录路径')
    parser.add_argument('--data_dir',
                       required=True,
                       help='原始数据目录路径（必需，用于计算真实的百分比准确率）')
    parser.add_argument('--scene_prefix',
                       default='V',
                       help='场景文件夹前缀，例如 V, A, K 等 (默认: V)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.excel_file):
        print(f"❌ Excel文件不存在: {args.excel_file}")
        print("请先运行 analyze_match_results.py 生成分析结果")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"❌ 数据目录不存在: {args.data_dir}")
        return
    
    print("🎯 使用真实数据计算百分比准确率...")
    print(f"📁 Excel文件: {args.excel_file}")
    print(f"📁 数据目录: {args.data_dir}")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🎯 场景前缀: {args.scene_prefix}")
    
    # 创建可视化器并运行
    visualizer = MatchResultVisualizer(args.excel_file, args.data_dir, args.scene_prefix)
    visualizer.run_visualization(args.output_dir)

if __name__ == "__main__":
    main()
