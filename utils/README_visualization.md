# 多模态匹配结果可视化工具 - 基于真实数据

## 功能特点

✨ **基于真实数据计算**
- 直接从原始匹配文件读取数据
- 使用RANSAC算法计算真实内点
- 按匹配百分比分段计算准确率
- 支持多场景、多模态配对数据聚合

🎨 **专业可视化**
- 热力图矩阵展示准确率分布
- 阈值曲线分析误差影响
- 百分比曲线分析匹配质量
- 3D交互图全方位展示
- 综合仪表板整合关键信息

🌍 **国际化支持**
- 所有图表标注已英文化
- 适合国际期刊发表
- 专业配色方案

## 使用方法

### 1. 安装依赖
```bash
pip install matplotlib seaborn plotly kaleido pandas numpy opencv-python openpyxl
```

### 2. 基本使用
```bash
python utils/visualize_match_results.py \
    --data_dir /path/to/original/data \
    --excel_file match_results_analysis.xlsx \
    --output_dir ./results/visualization_output
```

### 3. 快速演示
```bash
python demo_visualization.py
```

## 参数说明

- `--data_dir`: **必需** 原始数据目录路径，用于计算真实的百分比准确率
- `--excel_file`: Excel分析结果文件路径（默认: match_results_analysis.xlsx）
- `--output_dir`: 输出目录路径（默认: ./results/visualization_output）

## 输出文件

### 静态图片 (.png)
- `heatmap_accuracy_overview.png` - 主热力图矩阵
- `threshold_curves.png` - 阈值曲线图
- `percentage_curves.png` - 百分比曲线图  
- `modality_comparison.png` - 模态对比柱状图

### 交互式图表 (.html)
- `interactive_3d_plot.html` - 3D散点图
- `dashboard.html` - 综合分析仪表板

## 数据处理流程

1. **扫描数据目录** - 自动发现所有场景和模态配对
2. **读取匹配文件** - 加载排序后的匹配三元组(id1, id2, dist)
3. **计算坐标** - 将patch ID转换为图像坐标
4. **RANSAC处理** - 使用OpenCV计算单应性矩阵，过滤内点
5. **百分比分段** - 按10%, 20%, ..., 100%取前N个匹配
6. **准确率计算** - 对每个阈值(4,8,12,16,20像素)计算准确率
7. **数据聚合** - 跨场景和模态配对计算平均值
8. **图表生成** - 生成各种类型的可视化图表

## 注意事项

⚠️ **重要**: 此版本只支持真实数据计算，不再提供模拟数据选项，确保所有结果的真实性和可靠性。

📊 **性能**: 计算时间取决于数据规模，大型数据集可能需要较长时间处理。

🎯 **精度**: 所有准确率计算基于RANSAC算法的真实内点检测结果。