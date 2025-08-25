#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pseudo_multimodal3.py - 优化版伪多模态视频生成器
主要改进：
1. 时间平滑滤波减少闪烁
2. 新增多种运动敏感的模态变换
3. 增强运动信息保留
4. 更有效的视觉外观模糊
5. 自适应参数调整

用法示例：
python pseudo_multimodal3.py --input input.mp4 --output_dir ./out --resize 640 --variants all

依赖:
pip install opencv-python numpy scipy
"""
import os
import argparse
import cv2
import numpy as np
import time
from collections import deque
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# ---------- 辅助函数 ----------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def to_gray_float(frame):
    """BGR uint8 -> gray float [0,1]"""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return g

def temporal_smooth(buffer, sigma=1.0):
    """时间域高斯平滑"""
    if len(buffer) < 3:
        return buffer[-1]
    
    # 将buffer转换为numpy数组进行时间平滑
    stack = np.stack(buffer, axis=0)  # [T, H, W, C]
    smoothed = np.zeros_like(stack[-1])
    
    for c in range(stack.shape[-1]):
        for h in range(stack.shape[1]):
            for w in range(stack.shape[2]):
                time_series = stack[:, h, w, c]
                smoothed[h, w, c] = gaussian_filter1d(time_series, sigma=sigma)[-1]
    
    return smoothed.astype(np.uint8)

def adaptive_threshold(image, block_size=11, C=2):
    """自适应阈值处理"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, block_size, C)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

# ---------- 优化的变换函数 ----------
def enhanced_motion_thermal(prev_frame, cur_frame, motion_buffer, ksize=15, alpha=0.3, beta=0.7):
    """
    增强版运动热图 - 添加时间平滑和运动累积
    """
    prev_g = to_gray_float(prev_frame)
    cur_g = to_gray_float(cur_frame)
    
    # 计算运动差分
    diff = np.abs(cur_g - prev_g)
    
    # 运动累积 - 保留历史运动信息
    motion_buffer.append(diff)
    motion_accum = np.mean(motion_buffer, axis=0) if len(motion_buffer) > 1 else diff
    
    # 多尺度运动检测
    diff_blur1 = cv2.GaussianBlur((diff * 255).astype(np.uint8), (ksize, ksize), 0).astype(np.float32) / 255.0
    diff_blur2 = cv2.GaussianBlur((motion_accum * 255).astype(np.uint8), (ksize//2, ksize//2), 0).astype(np.float32) / 255.0
    
    # 结合当前和累积运动
    motion_combined = 0.6 * diff_blur1 + 0.4 * diff_blur2
    
    # 自适应归一化
    motion_norm = np.zeros_like(motion_combined)
    max_val = motion_combined.max()
    if max_val > 1e-6:
        motion_norm = motion_combined / (max_val + 1e-9)
    
    # 生成热图
    cur_blur = cv2.GaussianBlur((cur_g * 255).astype(np.uint8), (ksize, ksize), 0).astype(np.float32) / 255.0
    thermal = np.clip(alpha * cur_blur + beta * motion_norm, 0.0, 1.0)
    
    # 增强对比度
    thermal = np.power(thermal, 0.7)  # gamma校正增强对比度
    thermal_u8 = (thermal * 255).astype(np.uint8)
    thermal_color = cv2.applyColorMap(thermal_u8, cv2.COLORMAP_JET)
    
    return thermal_color

def temporal_gradient(frame_buffer, weights=None):
    """
    时间梯度模态 - 基于时间序列的梯度变化
    """
    if len(frame_buffer) < 3:
        return np.zeros_like(frame_buffer[-1])
    
    if weights is None:
        weights = [-1, 0, 1]  # 简单的时间梯度核
    
    # 转换为灰度
    gray_buffer = [to_gray_float(f) for f in frame_buffer[-3:]]
    
    # 计算时间梯度
    temporal_grad = np.zeros_like(gray_buffer[0])
    for i, w in enumerate(weights):
        temporal_grad += w * gray_buffer[i]
    
    # 归一化到[0,1]
    temporal_grad = np.abs(temporal_grad)
    max_val = temporal_grad.max()
    if max_val > 1e-6:
        temporal_grad = temporal_grad / max_val
    
    # 转换为伪彩色
    temporal_grad_u8 = (temporal_grad * 255).astype(np.uint8)
    colored = cv2.applyColorMap(temporal_grad_u8, cv2.COLORMAP_VIRIDIS)
    
    return colored

def motion_saliency(prev_frame, cur_frame, next_frame=None):
    """
    运动显著性模态 - 突出运动区域，模糊静态区域
    """
    prev_g = to_gray_float(prev_frame)
    cur_g = to_gray_float(cur_frame)
    
    # 计算前向和后向运动
    forward_motion = np.abs(cur_g - prev_g)
    
    if next_frame is not None:
        next_g = to_gray_float(next_frame)
        backward_motion = np.abs(next_g - cur_g)
        motion = (forward_motion + backward_motion) / 2
    else:
        motion = forward_motion
    
    # 运动显著性检测
    motion_blur = cv2.GaussianBlur((motion * 255).astype(np.uint8), (21, 21), 0).astype(np.float32) / 255.0
    
    # 自适应阈值确定运动区域
    motion_mask = motion_blur > (motion_blur.mean() + 0.5 * motion_blur.std())
    
    # 对静态区域应用强模糊
    result = cur_frame.copy().astype(np.float32)
    static_regions = cv2.GaussianBlur(cur_frame, (31, 31), 0).astype(np.float32)
    
    # 混合运动和静态区域
    for c in range(3):
        result[:, :, c] = np.where(motion_mask, result[:, :, c], static_regions[:, :, c])
    
    return np.clip(result, 0, 255).astype(np.uint8)

def frequency_domain_transform(frame, low_pass_ratio=0.1, high_pass_ratio=0.9):
    """
    频域变换模态 - 保留高频运动信息，模糊低频纹理
    """
    # 转换到频域
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    
    # 创建频域滤波器
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    
    # 高通滤波器 - 保留边缘和运动信息
    mask = np.ones((rows, cols), np.uint8)
    r_low = int(min(rows, cols) * low_pass_ratio)
    mask[crow-r_low:crow+r_low, ccol-r_low:ccol+r_low] = 0
    
    # 应用滤波器
    f_shift_filtered = f_shift * mask
    
    # 逆变换
    f_ishift = np.fft.ifftshift(f_shift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # 归一化
    img_back = ((img_back - img_back.min()) / (img_back.max() - img_back.min()) * 255).astype(np.uint8)
    
    return cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)

def structured_texture_removal(frame, structure_scale=1.0, texture_scale=0.1):
    """
    结构保留纹理移除 - 基于双边滤波的结构保留
    """
    # 多尺度双边滤波
    result = frame.copy()
    
    # 保留大尺度结构
    result = cv2.bilateralFilter(result, 15, 80, 80)
    result = cv2.bilateralFilter(result, 15, 80, 80)
    
    # 进一步模糊纹理
    result = cv2.medianBlur(result, 5)
    
    # 保留边缘信息
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 混合结果
    alpha = 0.85
    result = cv2.addWeighted(result, alpha, edges_colored, 1-alpha, 0)
    
    return result

def enhanced_optical_flow(prev_frame, cur_frame, flow_buffer, mag_mult=10.0):
    """
    增强光流可视化 - 添加时间平滑和流场累积
    """
    prev_g = (to_gray_float(prev_frame) * 255).astype(np.uint8)
    cur_g = (to_gray_float(cur_frame) * 255).astype(np.uint8)
    
    # 计算光流
    flow = cv2.calcOpticalFlowPyrLK(prev_g, cur_g, 
                                    corners=cv2.goodFeaturesToTrack(prev_g, maxCorners=1000, 
                                                                   qualityLevel=0.01, minDistance=10),
                                    nextPts=None)
    
    # Farneback光流作为补充
    flow_dense = cv2.calcOpticalFlowFarneback(prev_g, cur_g, None,
                                             pyr_scale=0.5, levels=3, winsize=15,
                                             iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    
    # 光流累积
    flow_buffer.append(flow_dense)
    if len(flow_buffer) > 1:
        flow_accumulated = np.mean(flow_buffer, axis=0)
    else:
        flow_accumulated = flow_dense
    
    # 计算幅值和角度
    mag, ang = cv2.cartToPolar(flow_accumulated[..., 0], flow_accumulated[..., 1])
    
    # 自适应幅值缩放
    mag_mean = mag.mean()
    adaptive_mult = mag_mult * (1.0 + mag_mean)
    
    # HSV可视化
    h = ang * 180 / np.pi / 2
    v = np.clip(mag * adaptive_mult, 0, 255)
    
    # 增强小运动的可见性
    v = np.power(v / 255.0, 0.5) * 255  # gamma校正
    
    hsv = np.zeros((flow_dense.shape[0], flow_dense.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.clip(h, 0, 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(v, 0, 255).astype(np.uint8)
    
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

# ---------- 主处理函数 ----------
def process_video_enhanced(input_path, output_dir,
                          resize_width=None,
                          variants=("enhanced_motion_thermal", "temporal_gradient", "motion_saliency", 
                                   "frequency_domain", "texture_removal", "enhanced_flow"),
                          temporal_smooth_sigma=1.0,
                          motion_buffer_size=5,
                          flow_buffer_size=3):
    """
    增强版视频处理函数
    """
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if resize_width is None:
        w, h = orig_w, orig_h
    else:
        w = int(resize_width)
        h = int(orig_h * resize_width / orig_w)

    # 初始化输出writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writers = {}
    for v in variants:
        out_path = os.path.join(output_dir, f"{v}.mp4")
        writers[v] = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writers[v].isOpened():
            raise IOError(f"Cannot open writer for {out_path}")

    # 初始化缓冲区
    frame_buffer = deque(maxlen=5)  # 用于时间梯度
    motion_buffer = deque(maxlen=motion_buffer_size)  # 用于运动累积
    flow_buffer = deque(maxlen=flow_buffer_size)  # 用于光流平滑
    output_buffers = {v: deque(maxlen=7) for v in variants}  # 用于时间平滑输出

    # 读取第一帧
    ret, prev = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Video contains no frames.")
    if (w, h) != (orig_w, orig_h):
        prev = cv2.resize(prev, (w, h), interpolation=cv2.INTER_LINEAR)

    frame_buffer.append(prev)

    # 写入第一帧
    for v in variants:
        if v in ["enhanced_flow", "temporal_gradient"]:
            writers[v].write(np.zeros_like(prev))  # 黑帧
        else:
            writers[v].write(prev.copy())

    idx = 1
    while True:
        ret, cur = cap.read()
        if not ret:
            break
        if (w, h) != (orig_w, orig_h):
            cur = cv2.resize(cur, (w, h), interpolation=cv2.INTER_LINEAR)

        frame_buffer.append(cur)
        
        # 准备下一帧（用于运动显著性）
        ret_next, next_frame = cap.read()
        if ret_next and (w, h) != (orig_w, orig_h):
            next_frame = cv2.resize(next_frame, (w, h), interpolation=cv2.INTER_LINEAR)

        # 生成各种模态
        results = {}
        
        if "enhanced_motion_thermal" in variants:
            start_time = time.time()
            results["enhanced_motion_thermal"] = enhanced_motion_thermal(prev, cur, motion_buffer)
            print(f"[DEBUG] Frame {idx}: Enhanced Motion Thermal computed. Time: {time.time() - start_time:.3f}s")

        if "temporal_gradient" in variants and len(frame_buffer) >= 3:
            start_time = time.time()
            results["temporal_gradient"] = temporal_gradient(frame_buffer)
            print(f"[DEBUG] Frame {idx}: Temporal Gradient computed. Time: {time.time() - start_time:.3f}s")

        if "motion_saliency" in variants:
            start_time = time.time()
            results["motion_saliency"] = motion_saliency(prev, cur, next_frame)
            print(f"[DEBUG] Frame {idx}: Motion Saliency computed. Time: {time.time() - start_time:.3f}s")

        if "frequency_domain" in variants:
            start_time = time.time()
            results["frequency_domain"] = frequency_domain_transform(cur)
            print(f"[DEBUG] Frame {idx}: Frequency Domain Transform computed. Time: {time.time() - start_time:.3f}s")

        if "texture_removal" in variants:
            start_time = time.time()
            results["texture_removal"] = structured_texture_removal(cur)
            print(f"[DEBUG] Frame {idx}: Structured Texture Removal computed. Time: {time.time() - start_time:.3f}s")

        if "enhanced_flow" in variants:
            start_time = time.time()
            results["enhanced_flow"] = enhanced_optical_flow(prev, cur, flow_buffer)
            print(f"[DEBUG] Frame {idx}: Enhanced Optical Flow computed. Time: {time.time() - start_time:.3f}s")

        # 应用时间平滑并写入
        for variant_name, result in results.items():
            output_buffers[variant_name].append(result)
            
            if len(output_buffers[variant_name]) >= 3:  # 有足够帧进行平滑
                start_time = time.time()
                smoothed = temporal_smooth(list(output_buffers[variant_name]), sigma=temporal_smooth_sigma)
                writers[variant_name].write(smoothed)
                print(f"[DEBUG] Frame {idx}: {variant_name} Temporal Smooth computed. Time: {time.time() - start_time:.3f}s")
            else:
                writers[variant_name].write(result)

        # 处理下一帧（如果存在）
        if ret_next:
            prev = cur
            cur = next_frame
        else:
            prev = cur

        idx += 1
        if idx % 50 == 0:
            print(f"[INFO] processed {idx} frames...")

    cap.release()
    for v in writers.values():
        v.release()
    print("[INFO] All enhanced outputs saved.")

# ---------- 命令行解析 ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate enhanced pseudo-multimodal videos from RGB video.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input video path")
    parser.add_argument("--output_dir", "-o", type=str, default="./out_enhanced", help="Output directory")
    parser.add_argument("--resize", type=int, default=None, help="Resize width (preserve aspect), optional")
    parser.add_argument("--variants", type=str, default="all",
                        help="Comma-separated variants: enhanced_motion_thermal,temporal_gradient,motion_saliency,frequency_domain,texture_removal,enhanced_flow or 'all'")
    parser.add_argument("--temporal_smooth", type=float, default=1.0, help="Temporal smoothing sigma")
    parser.add_argument("--motion_buffer", type=int, default=5, help="Motion accumulation buffer size")
    parser.add_argument("--flow_buffer", type=int, default=3, help="Optical flow buffer size")
    return parser.parse_args()

# ---------- 主入口 ----------
if __name__ == "__main__":
    args = parse_args()
    if args.variants.strip().lower() == "all":
        vars_list = ("enhanced_motion_thermal", "temporal_gradient", "motion_saliency", 
                    "frequency_domain", "texture_removal", "enhanced_flow")
    else:
        vars_list = tuple([v.strip() for v in args.variants.split(",") if v.strip() != ""])

    print(f"[INFO] Input: {args.input}")
    print(f"[INFO] Output dir: {args.output_dir}")
    print(f"[INFO] Enhanced variants: {vars_list}")
    print(f"[INFO] Temporal smoothing sigma: {args.temporal_smooth}")
    ensure_dir(args.output_dir)

    process_video_enhanced(args.input, args.output_dir, resize_width=args.resize,
                          variants=vars_list, temporal_smooth_sigma=args.temporal_smooth,
                          motion_buffer_size=args.motion_buffer, flow_buffer_size=args.flow_buffer)
