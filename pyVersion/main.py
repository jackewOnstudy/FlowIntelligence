import cv2
from fastdtw import fastdtw
import numpy as np
import pickle
from multiprocessing import Process
from time import sleep
from parameters import param_dict
from utils import *
from segment_match import *
import argparse

dataset_path = "/home/jackew/Project/FlowIntelligence/"
base_output_folder = "./TEST_OUTPUT"
 
motion_status_path = f"{base_output_folder}/MotionStatus"
motion_counts_path = f"{base_output_folder}/MotionCounts"
match_result_path = f"{base_output_folder}/MatchResult/List"
match_result_view_path = f"{base_output_folder}/MatchResult/Pictures"
match_result_video_path = f"{base_output_folder}/MatchResult/Videos"

match_thresholds_otsu_path = f"{base_output_folder}/MatchThresholdsOtsu"
# 确保输出目录存在
if not os.path.exists(match_thresholds_otsu_path):
    os.makedirs(match_thresholds_otsu_path)

ITERATE_TIMES = 4


# 参数设置，所有参数的默认值
parameters = param_dict['param_T10']
# Accessing the value of video_path from the dictionary
video_name1 = parameters['video_name1']
video_name2 = parameters['video_name2']
video_size1 = parameters['video_size1']
video_size2 = parameters['video_size2']

video_size1 = (video_size1[0] // 64 * 64, video_size1[1] // 64 * 64)
video_size2 = (video_size2[0] // 64 * 64, video_size2[1] // 64 * 64)

video_path1 = f"{dataset_path}/{video_name1}"
video_path2 = f"{dataset_path}/{video_name2}"
motion_threshold1 = parameters['motion_threshold1']
motion_threshold2 = parameters['motion_threshold2']

grid_size = parameters['grid_size']
grid_size2 = parameters['grid_size2']
i_stride1 = parameters['stride']
i_stride2 = parameters['stride2']

i_stride1 = (i_stride1[0] // 2,  i_stride1[1] // 2)
i_stride2 = (i_stride2[0] // 2, i_stride2[1] // 2)

shifting_flag = True if grid_size[0] != i_stride1[0] else False

is_global_otsu = False

# print(parameters)

def save_npy_files(video_path, motion_status_list, grid_size, output_path):
    file_name_with_ext = os.path.basename(video_path)
    file_name, file_ext = os.path.splitext(file_name_with_ext)
    file_folder = file_name[:-1] 
    side = file_name[-1]
    f_grid = f"{grid_size[0]}x{grid_size[1]}"
    output_folder = f"{output_path}/{file_folder}/{side}/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    outputfile = output_folder + f_grid + ".npy"
    np.save(outputfile, motion_status_list)
    print(f"{outputfile} Saved!")
    return 

def load_npy_files(video_path, grid_size, output_path):
    file_name_with_ext = os.path.basename(video_path)
    file_name, file_ext = os.path.splitext(file_name_with_ext)
    file_folder = file_name[:-1]
    side = file_name[-1]
    f_grid = f"{grid_size[0]}x{grid_size[1]}"
    loadfile = f"{output_path}/{file_folder}/{side}/{f_grid}.npy"
    # 从文件加载
    motion_status = np.load(loadfile)
    print(f"{loadfile} Loaded!")
    return motion_status

def save_match_result_list(video_path, match_result ,grid_size, output_path):
    file_name_with_ext = os.path.basename(video_path)
    file_name, file_ext = os.path.splitext(file_name_with_ext)
    file_folder = file_name[:-1]
    output_folder = f"{output_path}/{file_folder}"
    # print(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    f_grid = f"{grid_size[0]}x{grid_size[1]}"
    outputfile = f"{output_folder}/{f_grid}.npy"
    with open(outputfile, 'wb') as f:
        pickle.dump(match_result, f)
    print(f"{outputfile} Saved!")

def load_match_result_list(video_path, grid_size, output_path):
    file_name_with_ext = os.path.basename(video_path)
    file_name, file_ext = os.path.splitext(file_name_with_ext)
    file_folder = file_name[:-1]
    f_grid = f"{grid_size[0]}x{grid_size[1]}"
    loadfile = f"{output_path}/{file_folder}/{f_grid}.npy"
    with open(loadfile, 'rb') as f:
        match_result = pickle.load(f)
    print(f"{loadfile} Loaded!")
    return match_result

def process_triplets(triplets):
    used_p2 = set()  # 用于存储已经使用过的p2值
    result = []      # 存储最终结果
    p1_groups = {}   # 用于存储相同p1值的三元组

    # 第一遍遍历，按p1值分组
    for triplet in triplets:  # triplets已按dist排序
        p1 = triplet[0]
        if p1 not in p1_groups:
            p1_groups[p1] = []
        p1_groups[p1].append(triplet)

    # 按照每组最小dist的顺序处理
    # 因为triplets已按dist排序，每组的第一个元素就是该组最小的dist
    sorted_groups = sorted(p1_groups.values(), key=lambda group: group[0][2])

    # 处理每个p1组
    for triplets_group in sorted_groups:
        # 过滤掉p2已使用的三元组
        valid_triplets = [t for t in triplets_group if t[1] not in used_p2]
        if valid_triplets:
            # 因为原列表已按dist排序，取第一个即是dist最小的
            selected_triplet = valid_triplets[0]
            result.append(selected_triplet)
            used_p2.add(selected_triplet[1])

    return result  # 结果已经是按dist排序的


def remove_first_frames(motion_timestamps_per_grid):
    """
    移除每个网格时间序列的前5帧数据
    :param motion_timestamps_per_grid: 二维数组 (网格数, 帧数)
    :return: 处理后的新数组
    """
    # 检查输入有效性
    if len(motion_timestamps_per_grid) == 0 or len(motion_timestamps_per_grid[0]) <= 5:
        return motion_timestamps_per_grid.copy()
    
    # 创建新数组存储结果
    processed_data = []
    
    # 遍历每个网格的时间序列
    for grid_sequence in motion_timestamps_per_grid:
        # 使用切片操作移除前5帧 [5:]
        processed_sequence = grid_sequence[8:]
        processed_data.append(processed_sequence)
    
    return np.array(processed_data)


def remove_last_frames(motion_timestamps_per_grid):
    """
    移除每个网格时间序列的后5帧数据
    :param motion_timestamps_per_grid: 二维数组 (网格数, 帧数)
    :return: 处理后的新数组
    """
    # 检查输入有效性
    if len(motion_timestamps_per_grid) == 0 or len(motion_timestamps_per_grid[0]) <= 5:
        return motion_timestamps_per_grid.copy()
    
    # 创建新数组存储结果
    processed_data = []
    
    # 遍历每个网格的时间序列
    for grid_sequence in motion_timestamps_per_grid:
        # 使用切片操作移除后5帧 [:-5]
        processed_sequence = grid_sequence[:-8]
        processed_data.append(processed_sequence)
    
    return np.array(processed_data)

def cal_overlap_grid():
# ==================================这里是计算8x8基网格的count序列以及status序列================================================
    # 计算网格状态序列(值为运动像素个数)
    motion_count_per_grid1, num_cols1, num_rows1 = get_motion_count_with_shifting_grid_visualization(video_path1, i_stride1, grid_size, parameters=parameters)
    motion_count_per_grid2, num_cols2, num_rows2 = get_motion_count_with_shifting_grid_visualization(video_path2, i_stride2, grid_size2, parameters=parameters)

    # # # 自适应T1参数
    # motion_count_per_grid1, num_cols1, num_rows1 = get_motion_count_with_otsu(video_path1, i_stride1, grid_size, parameters=parameters)
    # motion_count_per_grid2, num_cols2, num_rows2 = get_motion_count_with_otsu(video_path2, i_stride2, grid_size2, parameters=parameters)

    # # 保存motion_count_per_grid
    save_npy_files(video_path1, motion_count_per_grid1, grid_size, motion_counts_path)
    save_npy_files(video_path2, motion_count_per_grid2, grid_size2, motion_counts_path)

    # 计算网格状态序列
    motion_status_per_grid1 = get_motion_status(motion_count_per_grid1, motion_threshold1[0])
    motion_status_per_grid2 = get_motion_status(motion_count_per_grid2, motion_threshold2[0])
    # print("motion_status_per_grid1.shape:", motion_status_per_grid1.shape)
    # print("motion_status_per_grid2.shape:", motion_status_per_grid2.shape)
    # print("motion_status_per_grid1:", np.sum(motion_status_per_grid1))
    # print("motion_status_per_grid2:", np.sum(motion_status_per_grid2))

    # video_name1_name, file_ext = os.path.splitext(video_name1)
    # video_name2_name, file_ext = os.path.splitext(video_name2)
    # if is_global_otsu:
    #     # 自适应T2参数 --2 全局自适应
    #     motion_status_per_grid1, threshold_otsu1 = get_motion_status_global_otsu(motion_count_per_grid1)
    #     motion_status_per_grid2, threshold_otsu2 = get_motion_status_global_otsu(motion_count_per_grid2)
    # else:
    #     # # 自适应T2参数 --1 网格内自适应
    #     motion_status_per_grid1, threshold_otsu1 = get_motion_status_with_otsu(motion_count_per_grid1)
    #     motion_status_per_grid2, threshold_otsu2 = get_motion_status_with_otsu(motion_count_per_grid2)
    # # 保存阈值到CSV
    # save_thresholds_to_csv(threshold_otsu1, f'{match_thresholds_otsu_path}/thresholds_{video_name1_name}_grid_{grid_size[0]}x{grid_size[1]}.csv', is_global_otsu)
    # save_thresholds_to_csv(threshold_otsu2, f'{match_thresholds_otsu_path}/thresholds_{video_name2_name}_grid_{grid_size[0]}x{grid_size[1]}.csv', is_global_otsu)

    save_npy_files(video_path1, motion_status_per_grid1, grid_size, motion_status_path)
    save_npy_files(video_path2, motion_status_per_grid2, grid_size2, motion_status_path)


#==========================================这里是由8x8网格迭代构造4N倍大网格================================================
    # ######################################## TEST ##################################################
    # num_cols1 = (video_size1[0] - grid_size[0]) // i_stride1[0] + 1
    # num_rows1 = (video_size1[1] - grid_size[1]) // i_stride1[1] + 1
    # num_cols2 = (video_size2[0] - grid_size2[0]) // i_stride2[0] + 1
    # num_rows2 = (video_size2[1] - grid_size2[1]) // i_stride2[1] + 1
    # # print(f"num_cols1: {num_cols1}, num_rows1: {num_rows1}")
    # # print(f"num_cols2: {num_cols2}, num_rows2: {num_rows2}")
    # # print(f"grid_size: {grid_size}, grid_size2: {grid_size2}")
    # # print(f"i_stride1: {i_stride1}, i_stride2: {i_stride2}")
    # motion_count_per_grid1 = load_npy_files(video_path1, grid_size, motion_counts_path)
    # motion_count_per_grid2 = load_npy_files(video_path2, grid_size2, motion_counts_path)
    # # print("motion_count_per_grid1:", np.sum(motion_count_per_grid1))
    # # print("motion_count_per_grid2:", np.sum(motion_count_per_grid2))
    # ########################################## deviation ##################################################
    # # motion_count_per_grid1 = remove_first_frames(motion_count_per_grid1)
    # # motion_count_per_grid2 = remove_last_frames(motion_count_per_grid2)
    # motion_status_per_grid1 = get_motion_status(motion_count_per_grid1, motion_threshold1[0])
    # motion_status_per_grid2 = get_motion_status(motion_count_per_grid2, motion_threshold2[0])
    # # print("motion_status_per_grid1.shape:", motion_status_per_grid1.shape)
    # # print("motion_status_per_grid2.shape:", motion_status_per_grid2.shape)
    # # print("motion_status_per_grid1:", np.sum(motion_status_per_grid1))
    # # print("motion_status_per_grid2:", np.sum(motion_status_per_grid2))
    
    # save_npy_files(video_path1, motion_status_per_grid1, grid_size, motion_status_path)
    # save_npy_files(video_path2, motion_status_per_grid2, grid_size2, motion_status_path)
    # ###################################### TEST ##################################################

    # 初始网格大小都是8x8，以此构造4xN倍网格状态序列
    temp_grid_size = grid_size
    temp_grid_size2 = grid_size2
    # shift_grid_size = stride1
    # shift_grid_size2 = stride2
    for i in range(1,ITERATE_TIMES):
        temp_grid_size = (temp_grid_size[0] * 2, temp_grid_size[1] * 2)
        temp_grid_size2 = (temp_grid_size2[0] * 2, temp_grid_size2[1] * 2)
        # print(f"Constructing {i}th iteration: Grid size: {temp_grid_size}::{temp_grid_size2}")
        # print(f"motion_count_per_grid1.shape: {motion_count_per_grid1.shape}")
        # print(f"motion_count_per_grid2.shape: {motion_count_per_grid2.shape}")
        # print("motion_count_per_grid1:", np.sum(motion_count_per_grid1))
        # print("motion_count_per_grid2:", np.sum(motion_count_per_grid2))
        motion_count_per_grid1, num_cols1, num_rows1 = get_4n_grid_motion_count(motion_count_per_grid1, num_cols1, num_rows1, shifting_flag)
        motion_count_per_grid2, num_cols2, num_rows2 = get_4n_grid_motion_count(motion_count_per_grid2, num_cols2, num_rows2, shifting_flag)
        # print(f"motion_count_per_grid1.shape: {motion_count_per_grid1.shape}")
        # print(f"motion_count_per_grid2.shape: {motion_count_per_grid2.shape}")
        # print("motion_count_per_grid1:", np.sum(motion_count_per_grid1))
        # print("motion_count_per_grid2:", np.sum(motion_count_per_grid2))

        motion_status_per_grid1 = get_motion_status(motion_count_per_grid1, motion_threshold1[i])
        motion_status_per_grid2 = get_motion_status(motion_count_per_grid2, motion_threshold2[i])
        # print("shape:", motion_status_per_grid1.shape)
        # print("motion_threshold1:", motion_threshold1[i])
        # print("motion_status_per_grid1:", np.sum(motion_status_per_grid1))
        # print("motion_status_per_grid2:", np.sum(motion_status_per_grid2))

        # if is_global_otsu:
        #     # 自适应T3参数 --2 全局自适应
        #     motion_status_per_grid1, threshold_otsu1 = get_motion_status_global_otsu(motion_count_per_grid1)
        #     motion_status_per_grid2, threshold_otsu2 = get_motion_status_global_otsu(motion_count_per_grid2)
        # else:
        #     # 自适应T3参数 --1 网格内自适应
        #     motion_status_per_grid1, threshold_otsu1 = get_motion_status_with_otsu(motion_count_per_grid1)
        #     motion_status_per_grid2, threshold_otsu2 = get_motion_status_with_otsu(motion_count_per_grid2)
        # # 保存阈值到CSV
        # save_thresholds_to_csv(threshold_otsu1, f'{match_thresholds_otsu_path}/thresholds_{video_name1_name}_grid_{temp_grid_size[0]}x{temp_grid_size[1]}.csv', is_global_otsu)
        # save_thresholds_to_csv(threshold_otsu2, f'{match_thresholds_otsu_path}/thresholds_{video_name2_name}_grid_{temp_grid_size2[0]}x{temp_grid_size2[1]}.csv', is_global_otsu)

        save_npy_files(video_path1, motion_status_per_grid1, temp_grid_size, motion_status_path)
        save_npy_files(video_path2, motion_status_per_grid2, temp_grid_size2, motion_status_path)


#==========================================这里是由大到小进行匹配================================================
    match_result = []
    sorted_large_grid_corre_small_dict = {}
    for iterate_time in range(ITERATE_TIMES):
        print(f"Matching {iterate_time}th iteration: Grid size: {temp_grid_size}::{temp_grid_size2}")
        motion_status_per_grid1 = load_npy_files(video_path1, temp_grid_size, motion_status_path)
        motion_status_per_grid2 = load_npy_files(video_path2, temp_grid_size2, motion_status_path)

        # 将长序列分为多个阶段进行匹配，每个阶段进行滑动窗口
        # motion_status_per_grid1_segments = segment_matrix(motion_status_per_grid1, parameters['stage_length'])
        # motion_status_per_grid2_segments = segment_matrix(motion_status_per_grid2, parameters['stage_length'])
        motion_status_per_grid1_segments = segment_matrix(motion_status_per_grid1, len(motion_status_per_grid1[0]))
        motion_status_per_grid2_segments = segment_matrix(motion_status_per_grid2, len(motion_status_per_grid2[0]))

        match_result_all = [] # 保存一个网格大小下所有匹配结果（match + Propagate）
        for idx, (motion_status_per_grid1, motion_status_per_grid2) in enumerate(zip(motion_status_per_grid1_segments, motion_status_per_grid2_segments)):
            # print(f"    Matching stage {idx + 1}")
            # 在长序列上进行滑动窗口
            if idx == 0:
                match_result = find_matching_grid_with_segment(motion_status_per_grid1, motion_status_per_grid2, parameters, sorted_large_grid_corre_small_dict, num_cols1, num_cols1 // 2, shifting_flag)
                # 最大网格匹配，只保留前N个匹配结果，认为是足够准确的匹配
                if iterate_time == 0:
                    match_result = match_result[:20 if len(match_result) >20 else len(match_result)]
                # print('Stage1 match_list',match_result)
                match_result = process_triplets(match_result)
                match_result_all.append(match_result)
                # 当长序列只有一个阶段时，最后需要propagate_matching_result
                if len(motion_status_per_grid1_segments) != 1:
                    print("Not Only one stage, continue")
                    continue
           
            # TODO 这里需要控制在哪些iterate阶段进行propagate_matching_result
            if True: # iterate_time == 0:
            # Propagate the last matching result to the end
                for i in range(6):
                    start_time = time.time()
                    match_result = propagate_matching_result(match_result, motion_status_per_grid1, motion_status_per_grid2, num_rows1, num_cols1, num_rows2, num_cols2 ,parameters, shifting_flag)
                    end_time = time.time()
                    foldername = "/mnt/mDisk/Project/CityCam/Output/citydata.txt"
                    with open(foldername, "a") as f:
                        print(f"Matching {iterate_time}th iteration: Grid size: {temp_grid_size}::{temp_grid_size2}",file=f)
                        print(f"    Propagate {i} time: {end_time - start_time}",file=f)
                    match_result = process_triplets(match_result)
                    match_result_all.append(match_result)
        # print(match_result)
        # 删除重复的匹配结果
        match_result = process_triplets(match_result)
        
        print(match_result[:16])
        stride1 = temp_grid_size
        stride2 = temp_grid_size2
        if shifting_flag:
            stride1 = (stride1[0] // 2, stride1[1] // 2)
            stride2 = (stride2[0] // 2, stride2[1] // 2)
        match_result_view(video_path1, match_result, temp_grid_size, stride1, match_result_view_path, 1)
        match_result_view(video_path2, match_result, temp_grid_size2, stride2, match_result_view_path, 2)
        save_match_result_list(video_path1, match_result_all, temp_grid_size, match_result_path)
        
        small_grid_size = (temp_grid_size[0] // 2, temp_grid_size[1] // 2)
        small_grid_size2 = (temp_grid_size2[0] // 2, temp_grid_size2[1] // 2)

        if shifting_flag:
            num_cols1, num_rows1 = num_cols1 * 2 + 1, num_rows1 * 2 + 1
            num_cols2, num_rows2 = num_cols2 * 2 + 1, num_rows2 * 2 + 1
        else:
            num_cols1, num_rows1 = num_cols1 * 2, num_rows1 * 2
            num_cols2, num_rows2 = num_cols2 * 2, num_rows2 * 2

        sorted_large_grid_corre_small_dict = get_small_grid_index(match_result, video_size2, small_grid_size2, temp_grid_size2, shifting_flag, 2)
        temp_grid_size = small_grid_size
        temp_grid_size2 = small_grid_size2


    
if __name__ == '__main__':
    match_result_all = cal_overlap_grid()