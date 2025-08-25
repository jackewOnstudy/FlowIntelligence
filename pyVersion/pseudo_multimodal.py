#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pseudo_multimodal.py
将单通道（RGB）视频转成多种“伪多模态”视频：
 - motion_thermal (运动能量热图)
 - event (事件帧 / 二值差分)
 - flow_color (光流方向/幅值伪彩色)
 - blur_posterize (强模糊 + 颜色量化)
 - edge (边缘图)

用法示例：
python pseudo_multimodal.py --input input.mp4 --output_dir ./out --resize 640 --variants all

依赖:
pip install opencv-python numpy
"""
import os
import argparse
import cv2
import numpy as np
from collections import deque

# ---------- 辅助函数 ----------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def to_gray_float(frame):
    """BGR uint8 -> gray float [0,1]"""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return g

def write_video(frames, out_path, fps):
    """frames: list of BGR uint8 frames"""
    if len(frames) == 0:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise IOError(f"Cannot open video writer for {out_path}")
    for f in frames:
        writer.write(f)
    writer.release()

# ---------- 变换函数 ----------
def motion_thermal(prev_frame, cur_frame, ksize=11, alpha=0.2, beta=0.8):
    """
    生成运动-热图风格帧
    prev_frame, cur_frame: BGR uint8
    返回 BGR uint8
    """
    prev_g = to_gray_float(prev_frame)
    cur_g = to_gray_float(cur_frame)
    diff = np.abs(cur_g - prev_g)
    diff_blur = cv2.GaussianBlur((diff * 255).astype(np.uint8), (ksize, ksize), 0).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur((cur_g * 255).astype(np.uint8), (ksize, ksize), 0).astype(np.float32) / 255.0
    # 归一化运动响应（按当前帧）
    maxval = diff_blur.max()
    if maxval < 1e-6:
        motion_norm = diff_blur
    else:
        motion_norm = diff_blur / (maxval + 1e-9)
    thermal = np.clip(alpha * blur + beta * motion_norm, 0.0, 1.0)
    thermal_u8 = (thermal * 255).astype(np.uint8)
    thermal_color = cv2.applyColorMap(thermal_u8, cv2.COLORMAP_JET)
    return thermal_color

def event_like(prev_frame, cur_frame, thr=0.03):
    """
    事件帧：二值化的差分
    返回灰度单通道的 uint8 (converted to BGR when writing)
    """
    prev_g = to_gray_float(prev_frame)
    cur_g = to_gray_float(cur_frame)
    diff = cur_g - prev_g
    evt = (np.abs(diff) > thr).astype(np.uint8) * 255
    evt_bgr = cv2.cvtColor(evt, cv2.COLOR_GRAY2BGR)
    return evt_bgr

def flow_color(prev_frame, cur_frame, mag_mult=8.0):
    """
    计算光流并可视化为伪彩色流图 (BGR)
    使用 Farneback 光流（OpenCV）
    """
    prev_g = (to_gray_float(prev_frame) * 255).astype(np.uint8)
    cur_g = (to_gray_float(cur_frame) * 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(prev_g, cur_g, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # HSV: H: angle, S: 255, V: mag normalized
    h = ang * 180 / np.pi / 2   # map [0,2pi] -> [0,180]
    v = np.clip(mag * mag_mult, 0, 255)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.clip(h, 0, 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(v, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def blur_posterize(frame, blur_ksize=21, levels=8):
    """
    强模糊 + 颜色量化 (posterize-like)
    frame: BGR uint8
    """
    blur = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), 0)
    # 量化每通道
    levels = max(2, int(levels))
    factor = 256 // levels
    poster = (blur // factor) * factor + factor // 2
    poster = np.clip(poster, 0, 255).astype(np.uint8)
    return poster

def edge_only(frame, canny_low=50, canny_high=150, dilate_iter=1):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    if dilate_iter > 0:
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
    bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return bgr

# ---------- 主处理函数 ----------
def process_video(input_path, output_dir,
                  resize_width=None,
                  variants=("motion_thermal", "event", "flow_color", "blur_posterize", "edge"),
                  event_thr=0.03,
                  event_accum_window=3):
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

    # 初始化 writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writers = {}
    for v in variants:
        out_path = os.path.join(output_dir, f"{v}.mp4")
        writers[v] = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writers[v].isOpened():
            raise IOError(f"Cannot open writer for {out_path}")

    ret, prev = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Video contains no frames.")
    if (w, h) != (orig_w, orig_h):
        prev = cv2.resize(prev, (w, h), interpolation=cv2.INTER_LINEAR)

    # 写第一帧（或空帧）
    for v in variants:
        if v == "flow_color":
            writers[v].write(np.zeros_like(prev))  # 黑帧
        else:
            writers[v].write(prev.copy())

    event_buffer = deque(maxlen=event_accum_window)

    idx = 1
    while True:
        ret, cur = cap.read()
        if not ret:
            break
        if (w, h) != (orig_w, orig_h):
            cur = cv2.resize(cur, (w, h), interpolation=cv2.INTER_LINEAR)

        if "motion_thermal" in variants:
            writers["motion_thermal"].write(motion_thermal(prev, cur))

        if "event" in variants:
            ev = event_like(prev, cur, thr=event_thr)
            event_buffer.append(ev[...,0])
            acc = np.zeros_like(ev[...,0], dtype=np.uint8)
            for a in event_buffer:
                acc = np.clip(acc.astype(np.int32) + (a>0).astype(np.int32), 0, 255).astype(np.uint8)
            acc = (acc > 0).astype(np.uint8) * 255
            writers["event"].write(cv2.cvtColor(acc, cv2.COLOR_GRAY2BGR))

        if "flow_color" in variants:
            writers["flow_color"].write(flow_color(prev, cur))

        if "blur_posterize" in variants:
            writers["blur_posterize"].write(blur_posterize(cur))

        if "edge" in variants:
            writers["edge"].write(edge_only(cur))

        prev = cur
        idx += 1
        if idx % 50 == 0:
            print(f"[INFO] processed {idx} frames...")

    cap.release()
    for v in writers.values():
        v.release()
    print("[INFO] All outputs saved.")


# ---------- 命令行解析 ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Generate pseudo-multimodal videos from an RGB video.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input video path")
    parser.add_argument("--output_dir", "-o", type=str, default="./out", help="Output directory")
    parser.add_argument("--resize", type=int, default=None, help="Resize width (preserve aspect), optional")
    parser.add_argument("--variants", type=str, default="all",
                        help="Comma-separated variants to produce. options: motion_thermal,event,flow_color,blur_posterize,edge or 'all'")
    parser.add_argument("--event_thr", type=float, default=0.03, help="Threshold for event-like diff (float 0..1)")
    parser.add_argument("--event_accum", type=int, default=3, help="Accumulation window for event-like")
    return parser.parse_args()

# ---------- 主入口 ----------
if __name__ == "__main__":
    args = parse_args()
    if args.variants.strip().lower() == "all":
        vars_list = ("motion_thermal", "event", "flow_color", "blur_posterize", "edge")
    else:
        vars_list = tuple([v.strip() for v in args.variants.split(",") if v.strip() != ""])

    print(f"[INFO] Input: {args.input}")
    print(f"[INFO] Output dir: {args.output_dir}")
    print(f"[INFO] Variants: {vars_list}")
    ensure_dir(args.output_dir)

    process_video(args.input, args.output_dir, resize_width=args.resize,
                  variants=vars_list, event_thr=args.event_thr, event_accum_window=args.event_accum)
