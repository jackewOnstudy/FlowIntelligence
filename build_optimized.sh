#!/bin/bash

# 视频匹配系统优化构建脚本
# Author: C++ Performance Optimization Expert
# 用于快速构建高性能版本

set -e  # 遇到错误立即退出

echo "=================================================="
echo "   视频匹配系统 - 高性能优化构建脚本"
echo "=================================================="
echo ""

# 检测系统信息
echo "🔍 检测系统信息..."
echo "操作系统: $(uname -s)"
echo "架构: $(uname -m)"
echo "CPU核心数: $(nproc)"

# 检测编译器
if command -v g++ >/dev/null 2>&1; then
    GCC_VERSION=$(g++ --version | head -n1)
    echo "GCC编译器: $GCC_VERSION"
fi

if command -v clang++ >/dev/null 2>&1; then
    CLANG_VERSION=$(clang++ --version | head -n1)
    echo "Clang编译器: $CLANG_VERSION"
fi

echo ""

# 检测依赖
echo "🔍 检测依赖库..."

# OpenCV检测
if pkg-config --exists opencv4; then
    OPENCV_VERSION=$(pkg-config --modversion opencv4)
    echo "✅ OpenCV: $OPENCV_VERSION"
elif pkg-config --exists opencv; then
    OPENCV_VERSION=$(pkg-config --modversion opencv)
    echo "✅ OpenCV: $OPENCV_VERSION"
else
    echo "❌ OpenCV未找到，请安装OpenCV开发库"
    echo "   Ubuntu/Debian: sudo apt-get install libopencv-dev"
    echo "   CentOS/RHEL: sudo yum install opencv-devel"
    exit 1
fi

# OpenMP检测
if command -v omp_get_num_threads >/dev/null 2>&1 || ldconfig -p | grep -q libgomp; then
    echo "✅ OpenMP支持"
else
    echo "⚠️  OpenMP未找到，将禁用并行化优化"
    echo "   Ubuntu/Debian: sudo apt-get install libomp-dev"
    echo "   CentOS/RHEL: sudo yum install libgomp-devel"
fi

echo ""

# 创建构建目录
echo "📁 准备构建目录..."
if [ -d "build" ]; then
    echo "清理现有构建目录..."
    rm -rf build
fi

mkdir build
cd build

echo ""

# 配置构建
echo "⚙️  配置构建系统..."
echo "构建类型: Release (最佳性能)"
echo "编译器优化: -O3 -march=native -mavx2 -flto"
echo "并行化: OpenMP"
echo "SIMD指令: AVX2, FMA, POPCNT"
echo ""

# 执行CMake配置
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG -march=native -mtune=native -mavx2 -mfma -mpopcnt -flto" \
      -DCMAKE_EXE_LINKER_FLAGS_RELEASE="-flto" \
      ..

if [ $? -ne 0 ]; then
    echo "❌ CMake配置失败"
    exit 1
fi

echo ""

# 编译
echo "🔨 开始编译..."
echo "使用 $(nproc) 个并行任务进行编译..."

make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "❌ 编译失败"
    exit 1
fi

echo ""
echo "✅ 编译成功!"
echo ""

# 显示结果
if [ -f "FlowIntelligence" ]; then
    echo "📋 构建结果:"
    echo "可执行文件: $(pwd)/FlowIntelligence"
    echo "文件大小: $(du -h FlowIntelligence | cut -f1)"
    echo ""
    
    # 检测CPU特性支持
    echo "🔍 CPU特性检测:"
    if grep -q avx2 /proc/cpuinfo; then
        echo "✅ AVX2指令集支持"
    else
        echo "⚠️  AVX2指令集不支持，性能可能受限"
    fi
    
    if grep -q fma /proc/cpuinfo; then
        echo "✅ FMA指令支持"
    fi
    
    if grep -q popcnt /proc/cpuinfo; then
        echo "✅ POPCNT指令支持"
    fi
    
    echo ""
    echo "🚀 运行建议:"
    echo "1. 基本运行:"
    echo "   ./FlowIntelligence --video1 video1.mp4 --video2 video2.mp4"
    echo ""
    echo "2. 优化运行 (设置CPU亲和性):"
    echo "   taskset -c 0-$(($(nproc)-1)) ./FlowIntelligence [参数]"
    echo ""
    echo "3. 设置OpenMP线程数:"
    echo "   export OMP_NUM_THREADS=$(nproc)"
    echo "   ./FlowIntelligence [参数]"
    echo ""
    echo "4. 完整优化运行:"
    echo "   export OMP_NUM_THREADS=$(nproc)"
    echo "   taskset -c 0-$(($(nproc)-1)) ./FlowIntelligence \\"
    echo "     --video1 video1.mp4 --video2 video2.mp4 \\"
    echo "     --output-path ./output \\"
    echo "     --max-frames 1000"
    echo ""
    echo "📖 详细文档请参考: ../OPTIMIZATION_SUMMARY.md"
    echo ""
    echo "=================================================="
    echo "   构建完成! 享受高性能视频匹配体验 🎯"
    echo "=================================================="
else
    echo "❌ 未找到可执行文件"
    exit 1
fi 