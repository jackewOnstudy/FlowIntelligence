# FlowIntelligence 纯净版

这是 FlowIntelligence 的纯净版本，仅包含原版视频匹配功能，去除了所有增强功能和额外依赖。

## 🎯 包含的功能

- **原版视频匹配**: 基于 patch 的视频对齐和匹配算法
- **时间对齐**: 视频时间同步功能
- **运动检测**: 基础的运动区域检测
- **分段匹配**: 视频分段处理和匹配

## 📦 依赖项

**必需依赖:**
- OpenCV 4.0+
- CMake 3.16+
- C++17 编译器

**可选依赖:**
- OpenMP (用于并行加速)

## 🔨 构建方法

### 方法1: 使用简化构建脚本 (推荐)

```bash
# 运行纯净版构建脚本
./build_simple.sh
```

### 方法2: 手动构建

```bash
# 创建构建目录
mkdir build_simple
cd build_simple

# 备份原CMakeLists.txt并使用简化版本
cp ../CMakeLists.txt ../CMakeLists.txt.backup
cp ../CMakeLists_simple.txt ../CMakeLists.txt

# 配置和编译
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 恢复原CMakeLists.txt
cp ../CMakeLists.txt.backup ../CMakeLists.txt
rm ../CMakeLists.txt.backup
```

## 🚀 使用方法

构建完成后，可执行文件位于 `build_simple/` 目录下：

```bash
cd build_simple

# 基本用法
./FlowIntelligence --video1 video1.mp4 --video2 video2.mp4 \
    --dataset-path /path/to/videos \
    --output-path /path/to/output

# 查看所有选项
./FlowIntelligence --help
```

## ⚙️  主要参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--video1` | 第一个视频文件名 | T0A.mp4 |
| `--video2` | 第二个视频文件名 | B201A.mp4 |
| `--dataset-path` | 视频文件所在目录 | - |
| `--output-path` | 输出结果目录 | - |
| `--grid-size` | 初始网格大小 | 8x8 |
| `--stride` | 步长 | 8x8 |
| `--segment-length` | 分段长度 | 10 |
| `--max-frames` | 最大处理帧数 | 1000 |
| `--enable-time-alignment` | 启用时间对齐 | false |

## 📁 输出结果

程序会在指定的输出目录下创建以下子目录：

- `MotionStatus/` - 运动状态结果
- `MotionCounts/` - 运动统计数据
- `MatchResult/List/` - 匹配结果列表
- `MatchResult/Pictures/` - 匹配结果可视化图片

## 🔧 故障排除

1. **OpenCV 找不到**: 确保安装了 OpenCV 4.0+ 并设置了正确的环境变量
2. **编译错误**: 检查 C++17 编译器支持
3. **运行时错误**: 确保视频文件路径正确且可访问

## 📝 与增强版的区别

纯净版去除了以下功能和依赖:
- ❌ CUDA GPU 加速
- ❌ FFTW 频域分析
- ❌ 增强特征提取器
- ❌ 质量评估器
- ❌ 双向匹配器
- ❌ 时间一致性强化器
- ❌ 集成测试和单元测试
- ❌ 示例程序

保留的核心功能:
- ✅ 基本视频匹配算法
- ✅ 时间对齐
- ✅ 运动检测
- ✅ OpenMP 并行加速 (可选)
