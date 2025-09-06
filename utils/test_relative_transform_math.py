#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相对透视变换数学计算验证脚本

验证相对透视变换计算的数学正确性
"""

import numpy as np
import cv2


def test_relative_transform_math():
    """测试相对透视变换的数学计算"""
    
    print("🧮 相对透视变换数学验证")
    print("="*50)
    
    # 模拟图像尺寸
    width, height = 640, 480
    
    # 创建测试用的透视变换矩阵
    # H1: 第一个模态的变换
    src_points1 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst_points1 = np.float32([[20, 15], [width-25, 10], [width-30, height-20], [15, height-25]])
    H1 = cv2.getPerspectiveTransform(src_points1, dst_points1)
    
    # H2: 第二个模态的变换  
    dst_points2 = np.float32([[10, 20], [width-15, 25], [width-20, height-15], [25, height-30]])
    H2 = cv2.getPerspectiveTransform(src_points1, dst_points2)
    
    print("第一个模态变换矩阵 H1:")
    print(H1)
    print(f"H1 条件数: {np.linalg.cond(H1):.2f}")
    
    print("\n第二个模态变换矩阵 H2:")
    print(H2)
    print(f"H2 条件数: {np.linalg.cond(H2):.2f}")
    
    # 计算相对变换: H_relative = H2 × H1^(-1)
    H1_inv = np.linalg.inv(H1)
    H_relative = np.dot(H2, H1_inv)
    
    print("\n相对变换矩阵 H_relative = H2 × H1^(-1):")
    print(H_relative)
    print(f"H_relative 条件数: {np.linalg.cond(H_relative):.2f}")
    print(f"H_relative 行列式: {np.linalg.det(H_relative):.4f}")
    
    # 验证计算正确性
    print("\n🔍 验证计算正确性:")
    
    # 创建测试点
    test_points = np.float32([[100, 100], [200, 150], [300, 200], [400, 300]])
    
    # 方法1: 直接变换
    # I -> modality1 -> modality2
    points_mod1 = cv2.perspectiveTransform(test_points.reshape(-1, 1, 2), H1).reshape(-1, 2)
    points_mod2_method1 = cv2.perspectiveTransform(points_mod1.reshape(-1, 1, 2), H_relative).reshape(-1, 2)
    
    # 方法2: 直接使用H2
    # I -> modality2  
    points_mod2_method2 = cv2.perspectiveTransform(test_points.reshape(-1, 1, 2), H2).reshape(-1, 2)
    
    # 计算差异
    diff = np.linalg.norm(points_mod2_method1 - points_mod2_method2, axis=1)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"测试点数量: {len(test_points)}")
    print(f"最大误差: {max_diff:.6f} 像素")
    print(f"平均误差: {mean_diff:.6f} 像素")
    
    if max_diff < 1e-10:
        print("✅ 数学计算验证通过！")
    else:
        print("❌ 数学计算存在误差")
    
    # 显示变换强度
    identity = np.eye(3)
    transform_distance = np.linalg.norm(H_relative - identity, 'fro')
    print(f"\n变换强度（Frobenius范数）: {transform_distance:.4f}")
    
    print("\n📊 详细验证结果:")
    for i, (p_orig, p_method1, p_method2, error) in enumerate(
        zip(test_points, points_mod2_method1, points_mod2_method2, diff)):
        print(f"点 {i+1}:")
        print(f"  原始: ({p_orig[0]:.1f}, {p_orig[1]:.1f})")
        print(f"  方法1: ({p_method1[0]:.4f}, {p_method1[1]:.4f})")
        print(f"  方法2: ({p_method2[0]:.4f}, {p_method2[1]:.4f})")
        print(f"  误差: {error:.8f}")
    
    return max_diff < 1e-10


def test_identity_matrix_case():
    """测试单位矩阵的情况"""
    
    print("\n" + "="*50)
    print("🆔 单位矩阵情况测试")
    print("="*50)
    
    # 创建单位矩阵和普通变换矩阵
    H_identity = np.eye(3, dtype=np.float32)
    
    width, height = 640, 480
    src_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst_points = np.float32([[15, 10], [width-20, 15], [width-25, height-10], [10, height-20]])
    H_transform = cv2.getPerspectiveTransform(src_points, dst_points)
    
    print("单位矩阵 (RGB模态):")
    print(H_identity)
    
    print("\n变换矩阵 (多模态):")
    print(H_transform)
    
    # 计算相对变换: RGB -> 多模态
    H_relative_1 = np.dot(H_transform, np.linalg.inv(H_identity))
    
    # 计算相对变换: 多模态 -> RGB  
    H_relative_2 = np.dot(H_identity, np.linalg.inv(H_transform))
    
    print("\nRGB -> 多模态的相对变换:")
    print(H_relative_1)
    print(f"应该等于变换矩阵本身: {np.allclose(H_relative_1, H_transform)}")
    
    print("\n多模态 -> RGB的相对变换:")
    print(H_relative_2)
    print(f"应该等于变换矩阵的逆: {np.allclose(H_relative_2, np.linalg.inv(H_transform))}")
    
    return True


if __name__ == "__main__":
    try:
        print("开始相对透视变换数学验证...")
        
        # 测试一般情况
        result1 = test_relative_transform_math()
        
        # 测试单位矩阵情况
        result2 = test_identity_matrix_case()
        
        print("\n" + "="*50)
        if result1 and result2:
            print("🎉 所有测试通过！相对变换计算数学正确。")
        else:
            print("❌ 部分测试失败，需要检查计算逻辑。")
        print("="*50)
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
