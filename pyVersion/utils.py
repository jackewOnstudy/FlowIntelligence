import cv2
from fastdtw import fastdtw
import numpy as np
from distance_metric import *
from time import sleep
import os
import time
from tqdm import tqdm
from skimage.filters import threshold_otsu
import pandas as pd

from parameters import param_dict

# parameters = param_dict['param_S1']

# 计算状态序列
def get_motion_count_with_otsu(video_path, stride, grid_size, parameters):
    """
    通过帧间差分进行运动检测，结合滑动窗口，
    T1 使用 Otsu 方法自适应计算。
    """
    GaussianBlurKernelSize = parameters['GaussianBlurKernel']
    # 不再直接使用 parameters['Binary_threshold']
    Motion_threshold1 = parameters['motion_threshold1']
    Motion_threshold2 = parameters['motion_threshold2']

    stride_w, stride_h = stride
    grid_w, grid_h = grid_size

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 视频信息
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_cols = (frame_w - grid_w) // stride_w + 1
    num_rows = (frame_h - grid_h) // stride_h + 1

    progress_bar = tqdm(total=total_frames // 2 - 1, desc=f"Processing {os.path.basename(video_path)}")

    # 读第一帧
    ret, frame = cap.read()
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, GaussianBlurKernelSize, 0)

    motion_timestamps_per_grid = []

    for _ in range(3000):
        # 隔帧读取
        ret, frame = cap.read()
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, GaussianBlurKernelSize, 0)

        diff = cv2.absdiff(prev_gray, gray)
        diff = cv2.GaussianBlur(diff, GaussianBlurKernelSize, 0)

        # —— 在这里用 Otsu 自动计算阈值 ——
        # 参数阈值设置为 0，最大值 1，类型 THRESH_BINARY + THRESH_OTSU
        _, bin_diff = cv2.threshold(
            diff, 0, 1,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        grid_motion = []
        for i in range(num_rows):
            for j in range(num_cols):
                x, y = j * stride_w, i * stride_h
                patch = bin_diff[y:y+grid_h, x:x+grid_w]
                motion_sum = np.sum(patch)
                grid_motion.append(motion_sum)

        motion_timestamps_per_grid.append(grid_motion)
        prev_gray = gray
        progress_bar.update(1)

    progress_bar.close()
    cap.release()
    cv2.destroyAllWindows()

    # 转置：每行对应一个网格的时间序列
    motion_timestamps_per_grid = np.transpose(np.array(motion_timestamps_per_grid))
    return motion_timestamps_per_grid, num_cols, num_rows

def get_motion_count_with_shifting_grid_visualization(video_path, stride, grid_size, parameters):
    """
    通过帧间差分进行运动检测，结合滑动窗口
    """
    GaussianBlurKernelSize = parameters['GaussianBlurKernel']
    Binary_threshold = parameters['Binary_threshold']  # T1 in paper
    Motion_threshold1 = parameters['motion_threshold1']
    Motion_threshold2 = parameters['motion_threshold2']

    stride_w, stride_h = stride # 步长
    grid_w, grid_h = grid_size # 网格大小

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    motion_timestamps_per_grid = []

    # 获取视频帧的宽和高和帧数
    frame_w = int(cap.get(3))
    frame_h = int(cap.get(4))

    frame_w = frame_w // 64 * 64
    frame_h = frame_h // 64 * 64

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算网格数量
    num_cols = (frame_w - grid_w) // stride_w + 1
    num_rows = (frame_h - grid_h) // stride_h + 1

    file_name_with_ext = os.path.basename(video_path)
    file_name, file_ext = os.path.splitext(file_name_with_ext)

    # 使用 tqdm 创建进度条
    progress_bar = tqdm(total=total_frames // 2- 1, desc="Processing"+ file_name +" frames")

    # 设置初始帧
    ret, frame = cap.read()
    prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray = cv2.GaussianBlur(prev_frame_gray, GaussianBlurKernelSize, 0)
    # 统计运动了几次
    activate_table = np.zeros((num_rows, num_cols))

    #while True:
    for i in range(3000):
        # 读两帧，隔一帧进行差分
        ret, frame = cap.read()
        ret, frame = cap.read()
        if not ret:
            break
        # 转换为灰度图像
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, GaussianBlurKernelSize, 0)

        # 计算当前帧和前一帧的差异
        frame_diff = cv2.absdiff(prev_frame_gray, frame_gray)

        frame_diff = cv2.GaussianBlur(frame_diff, GaussianBlurKernelSize, 0)
        # frame_diff = cv2.equalizeHist(frame_diff)
        # 帧间差分结果阈值
        _, frame_diff = cv2.threshold(frame_diff, Binary_threshold, 1, cv2.THRESH_BINARY) # threshold

        motion_timestamps_grid = []

        # 遍历每个小网格
        for i in range(num_rows):
            for j in range(num_cols):
                # 计算当前网格的坐标
                grid_x = j * stride_w
                grid_y = i * stride_h

                # 提取当前网格区域
                grid_region = frame_diff[grid_y:grid_y+grid_h, grid_x:grid_x+grid_w]

                # 计算区域内像素值变化的均值
                motion_mean = np.sum(grid_region) 

                # # 判断运动状态，这里的阈值主要是为了去抖动
                # if motion_mean > Motion_mean_threshold:
                #     # 画红色矩形框
                #     cv2.rectangle(frame, (grid_x, grid_y), (grid_x + grid_w, grid_y + grid_h), (255, 0, 0), 1)
                #     motion_timestamps_grid.append(1)
                #     activate_table[i][j] += 1
                # else:
                #     motion_timestamps_grid.append(0)

                motion_timestamps_grid.append(motion_mean)

                # # 在网格左上角显示activate值
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # font_scale = 0.4
                # font_thickness = 1
                # text = f"{i * num_cols + j}"
                # # text = f"{i * num_cols + j}|{activate_table[i][j]}"
                # text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                # text_position = (grid_x, grid_y + text_size[1])
                # cv2.putText(frame, text, text_position, font, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)
        # 将当前帧的网格运动状态保存到数组
        motion_timestamps_per_grid.append(motion_timestamps_grid)

        # 更新前一帧
        prev_frame_gray = frame_gray.copy()
        # 更新进度条
        progress_bar.update(1)

        # 显示帧
        # cv2.imshow("Motion Detection", frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 关闭进度条
    progress_bar.close()
    cap.release()
    cv2.destroyAllWindows()

    # Convert to a NumPy array
    array_2d = np.array(motion_timestamps_per_grid)
    # Transpose the array
    # 让每一子数组都是一个网格的状态序列
    motion_timestamps_per_grid = np.transpose(array_2d)
    # print("motion_timestamps_per_grid:", motion_timestamps_per_grid)

    return motion_timestamps_per_grid, num_cols, num_rows

def get_4n_grid_motion_count(motion_count_per_grid, col_grid_num, row_grid_num, shifting_flag):
    """
    由小网格的状态构建4倍大小的大网格的状态，小网格状态不能是通过滑动窗口得到，即彼此之间不相交。
    主要有两种构建方式：
    @method1： 大网格由4个小网格构成，大网格不滑动刚好相接。
    此时假定大网格标号为K，一行总网格数为row_gird_num，简记为n，大网格对应的左上角小网格标号为：m 
    
    Args:
        motion_status_per_gird: 二维numpy数组，每一个子数组为一个小网格的状态序列
        col_grid_num: 一列有多少个小网格
        row_grid_num: 一行有多少个小网格
        shifting_flag: 大网格是否进行滑动窗口
    """
    # 输入网格总数
    total_grids = col_grid_num * row_grid_num
    # 每个小网格状态序列长度
    seq_len = motion_count_per_grid.shape[1]
    # 获取数组实际大小
    actual_grid_count = motion_count_per_grid.shape[0]
    
    # if shifting_flag:
    #     # 滑动窗口方式构建大网格
    #     new_col_grid_num = (col_grid_num // 2 + 1) // 2 + 1
    #     new_row_grid_num = (row_grid_num // 2 + 1) // 2 + 1
    # else:
    #     # 不滑动方式构建大网格
    #     new_col_grid_num = col_grid_num // 2
    #     new_row_grid_num = row_grid_num // 2

    new_col_grid_num = col_grid_num // 2
    new_row_grid_num = row_grid_num // 2

    new_motion_status = []

    for k in range(new_col_grid_num * new_row_grid_num):
        # if shifting_flag:
        #     row_idx = k // new_col_grid_num
        #     col_idx = k % new_col_grid_num
        #     m = row_idx * col_grid_num + col_idx * 2
        # else:
        row_idx = (k // new_col_grid_num) * 2  # 计算大网格左上角的行号
        col_idx = (k % new_col_grid_num) * 2   # 计算大网格左上角的列号
        m = row_idx * col_grid_num + col_idx

        if shifting_flag:
            m1 = m
            m2 = m + 2
            m3 = m + col_grid_num * 2
            m4 = m + (col_grid_num + 1) * 2
        else:
            m1 = m
            m2 = m + 1
            m3 = m + col_grid_num
            m4 = m + col_grid_num + 1
        # print(f"m1:{m1}, m2:{m2}, m3:{m3}, m4:{m4}")

        # 确保所有索引都在数组的实际大小范围内
        if m4 < actual_grid_count:
            # 直接将4个小网格的0/1状态序列合并
            # combined_status = np.bitwise_or.reduce([motion_count_per_grid[m1],
            #                                          motion_count_per_grid[m2],
            #                                          motion_count_per_grid[m3],
            #                                          motion_count_per_grid[m4]])
            
            # 将4个小网格的运动元素个数相加，后续可以根据阈值判断是否有运动
            combined_status = np.sum([  motion_count_per_grid[m1],
                                        motion_count_per_grid[m2],
                                        motion_count_per_grid[m3],
                                        motion_count_per_grid[m4]], axis=0)
            new_motion_status.append(combined_status)

    return np.array(new_motion_status), new_col_grid_num, new_row_grid_num

def get_motion_status_with_otsu(motion_count_2d, min_thr=1):
    """
    对 2D motion_count 做自适应阈值，
    输入:
      motion_count_2d: np.ndarray, shape=(num_grids, seq_len)
    输出:
      motion_status: np.ndarray of 0/1, same shape as motion_count_2d
      thresholds: np.ndarray of shape (num_grids,), 每个网格对应的自适应阈值
    """
    num_grids, seq_len = motion_count_2d.shape
    motion_status = np.zeros_like(motion_count_2d, dtype=np.uint8)
    thresholds = np.zeros((num_grids,), dtype=float)

    for idx in range(num_grids):
        counts = motion_count_2d[idx]
        
        # 检查数据是否有效（避免全零或单一值）
        if np.max(counts) == np.min(counts):
            # 如果所有值都相同，直接使用最小阈值
            thr = min_thr
        else:
            # 确保数据类型兼容 - 转换为float64避免类型问题
            counts_safe = counts.astype(np.float64)
            
            # 用 Otsu 自动算阈值
            try:
                thr = threshold_otsu(counts_safe)
            except Exception as e:
                print(f"Warning: Otsu threshold failed for grid {idx}, using min_thr. Error: {e}")
                thr = min_thr
        
        thr = max(thr, min_thr)
        # 若用熵方法，改为：
        # thr = threshold_li(counts)

        # 二值化：大于阈值的判为 1（有运动），否则为 0
        motion_status[idx] = (counts > thr).astype(np.uint8)
        thresholds[idx] = thr

    return motion_status, thresholds

def get_motion_status_global_otsu(motion_count_2d, min_thr=1):
    """
    motion_count_2d: shape=(num_grids, seq_len)
    返回：
      motion_status: 0/1，同 shape
      thresholds: shape=(seq_len,), 每帧对应的全局阈值
    """
    num_grids, seq_len = motion_count_2d.shape
    motion_status = np.zeros_like(motion_count_2d, dtype=np.uint8)
    thresholds = np.zeros((seq_len,), dtype=float)

    # 对每一帧 t：
    for t in range(seq_len):
        counts = motion_count_2d[:, t]               # shape=(num_grids,)
        
        # 检查数据是否有效（避免全零或单一值）
        if np.max(counts) == np.min(counts):
            # 如果所有值都相同，直接使用最小阈值
            thr = min_thr
        else:
            # 确保数据类型兼容 - 转换为float64避免类型问题
            counts_safe = counts.astype(np.float64)
            
            # 全局 Otsu
            try:
                thr = threshold_otsu(counts_safe)
            except Exception as e:
                print(f"Warning: Global Otsu threshold failed for frame {t}, using min_thr. Error: {e}")
                thr = min_thr
        
        # 可选：给一个下限，避免太小
        thr = max(thr, min_thr)
        thresholds[t] = thr

        # 根据阈值标记所有网格
        motion_status[:, t] = (counts > thr).astype(np.uint8)

    return motion_status, thresholds

def get_motion_status(motion_count, motion_threshold):
    """
    根据运动元素个数，判断是否有运动
    """
    motion_status = np.where(motion_count > motion_threshold, 1, 0)
    return motion_status

def get_small_grid_index(match_result, image_size, small_grid_size, large_grid_size, shifting_flag, which_video):
    """
    根据大网格标号K计算对应的小网格标号。
    
    Args:
        match result: 匹配结果
        image_size: 图片大小 (image_width, image_height)
        small_grid_size: 小网格大小 (small_width, small_height)
        large_grid_size: 大网格大小 (large_width, large_height)
        which_video: 1 or 2, 1表示第一个视频，2表示第二个视频
        
    Returns:
        which_video == 1:
            TODO
        which_video == 2:
            sorted_large_grid_corre_small_dict: V1大网格对应匹配V2大网格划分的小网格集合, key为大网格标号，value为小网格标号集合，按大网格标号排序
    """
    # 计算小网格和大网格的数量
    num_small_grids_per_row = image_size[0] // small_grid_size[0]
    num_large_grids_per_row = image_size[0] // large_grid_size[0]
    if shifting_flag:
        num_small_grids_per_row = num_small_grids_per_row * 2 - 1
        num_large_grids_per_row = num_large_grids_per_row * 2 - 1

    # 计算对应的小网格的行和列
    n_width = large_grid_size[0] // small_grid_size[0]
    n_height = large_grid_size[1] // small_grid_size[1]

    # 计算小网格的最终索引
    if which_video == 1:
        # TODO 这里的逻辑看后续需求
        for (index1, index2, point) in match_result:
            small_grid_index_set = get_small_index_in_large(index1, num_large_grids_per_row, num_small_grids_per_row, n_width, n_height, shifting_flag)
        return
    elif which_video == 2:
        large_grid_corre_small_dict = dict()
        for (index1, index2, point) in match_result:
            small_grid_index_set = get_small_index_in_large(index2, num_large_grids_per_row, num_small_grids_per_row, n_width, n_height, shifting_flag)
            if index1 in large_grid_corre_small_dict.keys():
                large_grid_corre_small_dict[index1].update(small_grid_index_set)
            else:
                large_grid_corre_small_dict[index1] = small_grid_index_set
        sorted_large_grid_corre_small_dict = {key:large_grid_corre_small_dict[key] for key in sorted(large_grid_corre_small_dict.keys())}
        return sorted_large_grid_corre_small_dict

def get_small_index_in_large(K, num_large_grids_per_row, num_small_grids_per_row, n_width, n_height, shifting_flag):
    # 计算大网格的行和列
    large_grid_row = K // num_large_grids_per_row
    large_grid_col = K % num_large_grids_per_row

    small_grid_row = large_grid_row * n_height
    small_grid_col = large_grid_col * n_width  
    if shifting_flag:
        small_grid_row = small_grid_row - 1
        small_grid_col = small_grid_col - 1

        n_height += 1
        n_width += 1

    small_grid_set = set()
    # 计算小网格的最终索引
    for i in range(n_height):
        for j in range(n_width):
            small_grid_index = (small_grid_row + i) * num_small_grids_per_row + (small_grid_col + j)
            small_grid_set.add(small_grid_index)

    return small_grid_set
    
def match_result_view(path, best_match_list, grid_sise, stride, output_path, which_video):
    stride_w, stride_h = stride
    grid_w, grid_h = grid_sise

    folder, file = os.path.split(path)
    first_frame_dir = folder + '/first_frame'
    file = file.split('.')[0]
    frame_path = first_frame_dir + '/' + file + '.jpg'

    frame = cv2.imread(frame_path)
    if frame is None:
        print("Error: Could not open frame.")
        return
    frame_w = frame.shape[1]
    frame_h = frame.shape[0]
    frame_w = frame_w // 64 * 64
    frame_h = frame_h // 64 * 64
    # print(f"frame_w:{frame_w}, frame_h:{frame_h}")

    num_cols = (frame_w - grid_w) // stride_w + 1
    num_rows = (frame_h - grid_h) // stride_h + 1

    for match in best_match_list:
        index1, index2, point = match
        # print(f"index1:{index1}, index2:{index2}, point:{point}")
        point = 0.5
        if which_video == 1:
            i1 = index1 // num_cols
            j1 = index1 % num_cols
            grid_x = j1 * stride_w  #change 
            grid_y = i1 * stride_h  #change
            # print(f"grid_x:{grid_x}, grid_y:{grid_y}")
            # print(f"i1:{i1}, j1:{j1},stride_w:{stride_w}, stride_h:{stride_h},num_cols:{num_cols}, num_rows:{num_rows}")
            # 加入红色滤镜
            red_mask = np.ones_like(frame[grid_y:grid_y+grid_h, grid_x:grid_x+grid_w]) * [0, 0, 255]
            # print(f"grid_x:{grid_x}, grid_y:{grid_y}, grid_w:{grid_w}, grid_h:{grid_h}")
            # print(f"frame[grid_y:grid_y+grid_h, grid_x:grid_x+grid_w].shape:{frame[grid_y:grid_y+grid_h, grid_x:grid_x+grid_w].shape}")
            # print(f"red_mask.shape:{red_mask.shape}")
            frame[grid_y:grid_y+grid_h, grid_x:grid_x+grid_w] = cv2.addWeighted(frame[grid_y:grid_y+grid_h, grid_x:grid_x+grid_w].astype(np.float32), 1 - point, red_mask.astype(np.float32), point, 0)

        if which_video == 2:
            i2 = index2 // num_cols
            j2 = index2 % num_cols
            grid_x = j2 * stride_w
            grid_y = i2 * stride_h
            # 加入绿色滤镜
            blue_mask = np.ones_like(frame[grid_y:grid_y+grid_h, grid_x:grid_x+grid_w]) * [0, 255, 0]
            frame[grid_y:grid_y+grid_h, grid_x:grid_x+grid_w] = cv2.addWeighted(frame[grid_y:grid_y+grid_h, grid_x:grid_x+grid_w].astype(np.float32), 1 - point, blue_mask.astype(np.float32), point, 0)
    
    # =========================================遍历每个小网格，标号===============================================================
    # for i in range(num_rows):
    #     for j in range(num_cols):
    #         # 计算当前网格的坐标
    #         grid_x = j * stride_w
    #         grid_y = i * stride_h
            
    #         # 在网格左上角显示标号
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         font_scale = 0.4
    #         font_thickness = 1
    #         text = f"{i * num_cols + j}"  # Calculate the grid index
    #         text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    #         text_position = (grid_x, grid_y + text_size[1])
    #         cv2.putText(frame, text, text_position, font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    # ========================================================================================================================

    output_folder = f"{output_path}/{file}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = f"{output_folder}/{grid_sise[0]}x{grid_sise[1]}_match_result.jpg"
    cv2.imwrite(output_file, frame)
    print(f"Save the match result to {output_file}")

    return

def isSurronded(match1_grid, match2_grid, num_cols, num_rows, radius=1):
    X1 = match1_grid // num_cols
    Y1 = match1_grid % num_cols
    X2 = match2_grid // num_cols
    Y2 = match2_grid % num_cols
    if abs(X1 - X2) <= radius and abs(Y1 - Y2) <= radius:
        return True 
    return False

def cross_validation(best_match_list_1, best_match_list_2, num_cols1, num_rows1,  num_cols2, num_rows2,radius = 1):
    """
    Perform cross-validation on two lists of best matches.
    
    将从1中找2的匹配与从2中找1的匹配得到的best_match_list交叉验证，如果两个匹配的结果是相邻的则认为正确。

    Args:
        best_match_list_1 (list): The first list of best matches.
        best_match_list_2 (list): The second list of best matches.
        num_cols (int): The number of columns in the frame, the basic unit is grid.
        num_rows (int): The number of rows.

    Returns:
        list: A list of best matches that satisfy the conditions.

    """
    best_match = []
    for match1 in best_match_list_1:
        for match2 in best_match_list_2:
            if match1 == match2:
                best_match.append(match1)
            elif match1[0] == match2[0] and isSurronded(match1[1], match2[1], num_cols=num_cols2, num_rows=num_rows2, radius = radius):
                best_match.append(match1)
                best_match.append(match2)
            elif match1[1] == match2[1] and isSurronded(match1[0], match2[0], num_cols=num_cols1, num_rows=num_rows1, radius = radius):
                best_match.append(match1)
                best_match.append(match2)

    return list(set(best_match))

def save_thresholds_to_csv(thresholds, filename, is_global_otsu):
    """
    保存阈值到CSV文件，包含统计信息
    根据阈值数组的维度自动判断是全局自适应还是网格内自适应
    
    Args:
        thresholds: 阈值数组
            - 全局自适应: shape为(seq_len,)
            - 网格内自适应: shape为(num_patches,)
        filename: 保存的文件名
    """
    import pandas as pd
    import numpy as np
    
    # 创建统计信息
    stats = {
        'mean': np.mean(thresholds),
        'std': np.std(thresholds),
        'min': np.min(thresholds),
        'max': np.max(thresholds),
        'median': np.median(thresholds),
        'q1': np.percentile(thresholds, 25),
        'q3': np.percentile(thresholds, 75)
    }
    
    # 创建DataFrame
    df_stats = pd.DataFrame([stats])
    df_thresholds = pd.DataFrame(thresholds, columns=['threshold'])
    
    is_global = is_global_otsu
    
    # 保存到CSV
    with open(filename, 'w') as f:
        f.write("# 阈值统计信息\n")
        df_stats.to_csv(f, index=False)
        f.write(f"\n# {'全局' if is_global else '网格内'}自适应阈值数据\n")
        df_thresholds.to_csv(f, index=True, index_label='frame_index' if is_global else 'patch_index')