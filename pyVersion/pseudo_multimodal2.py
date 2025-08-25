import cv2
import numpy as np
import os
import argparse
import sys

def generate_motion_gradient_frame(gray, prev_gray, prev_prev_gray):
    """
    通过三帧差分法生成运动梯度帧。
    这种方法能有效突出运动区域，抑制静止背景。
    """
    # 检查是否有足够的历史帧
    if prev_gray is None or prev_prev_gray is None:
        return np.zeros_like(gray)

    # 计算前后帧的差异
    diff1 = cv2.absdiff(gray, prev_gray)
    diff2 = cv2.absdiff(prev_gray, prev_prev_gray)
    
    # 对差异图像进行二值化处理，以消除噪声
    _, diff1_thresh = cv2.threshold(diff1, 25, 255, cv2.THRESH_BINARY)
    _, diff2_thresh = cv2.threshold(diff2, 25, 255, cv2.THRESH_BINARY)

    # 对两幅差异图像进行“与”操作，找到共同的运动区域
    motion_mask = cv2.bitwise_and(diff1_thresh, diff2_thresh)
    
    return motion_mask

def generate_edge_map_frame(frame):
    """
    使用Canny算法生成视频帧的边缘图。
    这种方法移除了纹理和颜色，只保留了场景的结构轮廓。
    """
    # 将帧转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用高斯模糊减少噪声，这有助于Canny算法获得更清晰的边缘
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 应用Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

def generate_phase_only_frame(frame):
    """
    通过傅里葉變換生成纯相位视频帧。
    这种方法保留了场景的结构和位置信息，但完全破坏了其视觉外观。
    """
    # 将帧转换为灰度图并转为浮点数类型
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)

    # 执行二维离散傅里叶变换
    fft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    # 将零频率分量移到频谱中心
    fft_shift = np.fft.fftshift(fft)

    # 分离幅度和相位
    magnitude, phase = cv2.cartToPolar(fft_shift[:, :, 0], fft_shift[:, :, 1])

    # 创建一个新的频谱，使用原始相位，但幅度全部设为1
    new_real, new_imag = cv2.polarToCart(np.ones_like(magnitude), phase)
    new_fft_shift = cv2.merge([new_real, new_imag])

    # 将零频率分量移回左上角
    new_fft = np.fft.ifftshift(new_fft_shift)
    # 执行逆傅里叶变换
    inverse_fft = cv2.idft(new_fft)

    # 取逆变换结果的幅度，并进行归一化处理
    result = cv2.magnitude(inverse_fft[:, :, 0], inverse_fft[:, :, 1])
    cv2.normalize(result, result, 0, 255, cv2.NORM_MINMAX)
    
    return np.uint8(result)

def generate_phase_only_frame_enhanced_visibility(frame):
    """
    通过傅里叶变换生成纯相位视频帧，并使用对数缩放增强可视化效果。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_float = np.float32(gray)

    fft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    fft_shift = np.fft.fftshift(fft)

    magnitude, phase = cv2.cartToPolar(fft_shift[:, :, 0], fft_shift[:, :, 1])

    # 关键步骤：幅度设为1
    new_real, new_imag = cv2.polarToCart(np.ones_like(magnitude), phase)
    new_fft_shift = cv2.merge([new_real, new_imag])

    new_fft = np.fft.ifftshift(new_fft_shift)
    inverse_fft = cv2.idft(new_fft)

    result = cv2.magnitude(inverse_fft[:, :, 0], inverse_fft[:, :, 1])

    # --- 修改部分：使用对数缩放增强可视性 ---
    # 1. 对结果应用对数函数。np.log1p 等价于 np.log(1 + x)，可以避免对0取对数的错误。
    result_log_scaled = np.log1p(result)
    
    # 2. 对经过对数缩放的结果进行归一化，而不是对原始结果进行归一化。
    cv2.normalize(result_log_scaled, result_log_scaled, 0, 255, cv2.NORM_MINMAX)
    
    return np.uint8(result_log_scaled)

def generate_posterize_nonlinear_frame(frame):
    """
    对视频帧进行极端色调分离和非线性色彩映射。
    这种方法破坏了局部的梯度和纹理，制造了误导性的视觉特征。
    """
    # 1. 极端色调分离 (Posterization)
    # 将每个颜色通道的256个级别量化为4个级别
    K = 4
    div = 256 / K
    # 避免被除数为0的警告
    with np.errstate(divide='ignore', invalid='ignore'):
        posterized = (frame.astype(np.float32) // div) * div + div // 2
    posterized = np.uint8(posterized)

    # 2. 非线性色彩映射 (Non-linear Mapping)
    # 应用正弦函数扭曲像素强度
    posterized_float = posterized.astype(np.float32)
    # 将像素值从[0, 255]映射到[0, 2*pi]，应用sin，再映射回[0, 255]
    nonlinear = (np.sin(posterized_float / 255.0 * 2 * np.pi) + 1) / 2 * 255
    
    return np.uint8(nonlinear)

def main(input_path):
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入视频文件未找到 '{input_path}'")
        sys.exit(1)

    # 创建输出目录
    output_dir = "./output_videos"
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出将保存到 '{output_dir}/' 目录中...")

    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("错误: 无法打开视频文件。")
        sys.exit(1)

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 定义视频编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 使用'mp4v'编码器以获得良好的兼容性
    
    # 定义输出文件路径
    out_paths = {
        'motion': os.path.join(output_dir, 'output_motion_gradient.mp4'),
        'edge': os.path.join(output_dir, 'output_edge_map.mp4'),
        'phase': os.path.join(output_dir, 'output_phase_only.mp4'),
        'posterize': os.path.join(output_dir, 'output_posterize_nonlinear.mp4'),
        'phase_enhanced': os.path.join(output_dir, 'output_phase_only_enhanced.mp4')
    }

    # 创建VideoWriter对象
    # 注意：即使处理结果是单通道灰度图，也需要转换为3通道BGR格式才能写入视频
    out_writers = {
        'motion': cv2.VideoWriter(out_paths['motion'], fourcc, fps, (width, height)),
        'edge': cv2.VideoWriter(out_paths['edge'], fourcc, fps, (width, height)),
        'phase': cv2.VideoWriter(out_paths['phase'], fourcc, fps, (width, height)),
        'posterize': cv2.VideoWriter(out_paths['posterize'], fourcc, fps, (width, height)),
        'phase_enhanced': cv2.VideoWriter(out_paths['phase_enhanced'], fourcc, fps, (width, height))
    }

    # 初始化用于帧差法的历史帧
    prev_gray, prev_prev_gray = None, None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"\r正在处理: 帧 {frame_count}/{total_frames}", end="")

        # 将当前帧转换为灰度图，供需要的方法使用
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. 生成运动梯度帧
        motion_frame = generate_motion_gradient_frame(current_gray, prev_gray, prev_prev_gray)
        out_writers['motion'].write(cv2.cvtColor(motion_frame, cv2.COLOR_GRAY2BGR))

        # 2. 生成边缘图帧
        edge_frame = generate_edge_map_frame(frame)
        out_writers['edge'].write(cv2.cvtColor(edge_frame, cv2.COLOR_GRAY2BGR))

        # 3. 生成纯相位帧
        phase_frame = generate_phase_only_frame(frame)
        out_writers['phase'].write(cv2.cvtColor(phase_frame, cv2.COLOR_GRAY2BGR))

        # 4. 生成色调分离与非线性映射帧
        posterize_frame = generate_posterize_nonlinear_frame(frame)
        out_writers['posterize'].write(posterize_frame)

        # 5. 生成增强可视性的纯相位帧
        phase_enhanced_frame = generate_phase_only_frame_enhanced_visibility(frame)
        out_writers['phase_enhanced'].write(cv2.cvtColor(phase_enhanced_frame, cv2.COLOR_GRAY2BGR))

        # 更新历史帧
        prev_prev_gray = prev_gray
        prev_gray = current_gray

    # 释放所有资源
    print("\n处理完成。正在释放资源...")
    cap.release()
    for writer in out_writers.values():
        writer.release()
    
    print("\n所有视频已成功生成！")
    for name, path in out_paths.items():
        print(f"- {name.capitalize()}: {path}")


if __name__ == '__main__':
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="""
        为特征匹配研究生成合成的“近似多模态”视频。
        该脚本接收一个输入视频，并生成四个经过不同算法变换的输出视频，
        这些变换旨在保留运动信息，同时破坏或挑战基于外观的匹配方法。
        """
    )
    parser.add_argument(
        "input_video", 
        help="输入视频文件的路径。"
    )
    args = parser.parse_args()
    
    main(args.input_video)