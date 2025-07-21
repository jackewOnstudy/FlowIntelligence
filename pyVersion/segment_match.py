import numpy as np
import time
from scipy.spatial.distance import hamming
from distance_metric import *
from tqdm import tqdm
from utils import get_small_index_in_large

def segment_sequence(sequence, segment_length):
    """将序列分段"""
    return [sequence[i:i + segment_length] for i in range(0, len(sequence), segment_length)]

def segment_matrix(matrix, segment_length):
    return [matrix[:, i:i + segment_length] for i in range(0, matrix.shape[1], segment_length)]

def segment_similarity(segment1, segment2, distance_metric):
    """计算两个段之间的距离"""
    if distance_metric == "levenshtein":
        seq1 = ''.join(map(str, segment1))
        seq2 = ''.join(map(str, segment2))
        return Levenshtein.distance(seq1, seq2)
    if distance_metric == "hamming":
        return hamming(segment1, segment2)
    if distance_metric == "logic_and":
        # and之后的距离比较大，因为状态序列是稀疏的（1元素本来就比较少）
        return logic_and_distance(segment1, segment2)
    if distance_metric == "logic_xonr":
        return logic_xonr_distance(segment1, segment2)
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

def find_matching_grid_with_segment(motion_status_matrix1, motion_status_matrix2, parameters, sorted_large_grid_corre_small_dict, small_grid_cols, large_grid_cols, shifting_flag):
    """
    在两个矩阵中找到匹配的网格
    Args:
        motion_status_matrix1: 第一个视角下的网格状态矩阵
        motion_status_matrix2: 第二个视角下的网格状态矩阵
        segment_length: 匹配时使用的序列段的长度
        max_mismatches: 最多不匹配段次数
    Return:
        列表, 元素为(grid1, grid2, dist)
    """
    grid_num1, seq_len1 = motion_status_matrix1.shape
    grid_num2, seq_len2 = motion_status_matrix2.shape

    assert seq_len1 == seq_len2, "两个矩阵的网格状态序列长度应相同"
    
    segment_length = parameters['segment_length']
    max_mismatches = parameters['max_mismatches']
    distance_metric = parameters['distance_metric']
    select_grid_factor = parameters['select_grid_factor']
    mismatch_distance_factor = parameters['mismatch_distance_factor']

    grid_list_v1 = []
    candidates = [ [] for i in range(grid_num1)]

    start_time = time.time()
    # 初始化候选集合
    #print("sorted_large_grid_corre_small_dict:", sorted_large_grid_corre_small_dict)
    if not sorted_large_grid_corre_small_dict:
        # 这里one_threshold越大，排除掉的候选网格越多【运动元素过少，不参与匹配】
        # one_threshold越小越合适，因为之前运动过滤已经过滤掉了应该认为无运动的网格
        one_threshold = len(motion_status_matrix1[0]) // select_grid_factor
        # print("one_threshold:", one_threshold)
        # print("grid_num1:", grid_num1)
        # print("range(grid_num1):", range(grid_num1))
        for i in range(grid_num1):
            if np.sum(motion_status_matrix1[i]) >= one_threshold:
                grid_list_v1.append(i)
                #print("grid_list_v1111:", grid_list_v1)
                valid_candidates = [j for j in range(grid_num2) if np.sum(motion_status_matrix2[j]) >= one_threshold]  ##################
                candidates[i]= valid_candidates
    else:
        # 通过控制两个缩放因子，放宽匹配条件
        max_mismatches = max_mismatches - 1 if max_mismatches - 1 > 1 else 1
        mismatch_distance_factor = mismatch_distance_factor - 2 if mismatch_distance_factor - 2 > 2 else 2
        
        for key in sorted_large_grid_corre_small_dict.keys():
            # ##### TEST #####
            # if key == 2117 or key == 2118:
            #     print("Exist")
            # ##### TEST #####
            v1_small_grid_set = get_small_index_in_large(key, large_grid_cols, small_grid_cols, 2, 2, shifting_flag)
            #print("v1_small_grid_set:", v1_small_grid_set)
            # 按我们每次按2倍关系进行递归，v1_small_grid_set中的元素有4个
            for i in list(v1_small_grid_set):
                # print("i:", i)
                if np.sum(motion_status_matrix1[i]) == 0:
                    # print("motion_status_matrix1[i]:", motion_status_matrix1[i])
                    v1_small_grid_set.remove(i)
                    # print("enter remove")
                else:
                    candidates[i] = [j for j in sorted_large_grid_corre_small_dict[key] if np.sum(motion_status_matrix2[j]) != 0] # 过滤掉没有运动的网格
                    # print("candidates[i]:", candidates[i])
            grid_list_v1.extend(item for item in v1_small_grid_set)
            #print("grid_list_v1222:", grid_list_v1)
            

    
    # print("grid_list_v1:", grid_list_v1)
    
    # 初始化不匹配段次数
    mismatch_counts = [np.zeros(grid_num2, dtype=int) for _ in range(grid_num1)]
    
    # 将所有网格的状态序列分段
    # segment1变为3维矩阵
    segments1 = [segment_sequence(motion_status_matrix1[i], segment_length) for i in range(grid_num1)]
    segments2 = [segment_sequence(motion_status_matrix2[j], segment_length) for j in range(grid_num2)]
    
    num_segments = len(segments1[0])

    # 创建进度条
    # total_steps = num_segments * len(grid_list_v1)
    # progress_bar = tqdm(total=total_steps, desc="Matching segments")

    # log_file = open("/home/jackew/Project/CityCam/Output/SegmentMatchLog.txt", 'w+')


    first_segment = True
    threshold_check_set = set()
    # print("num_segments:", num_segments)
    for segment_index in range(num_segments):
        #print("segment_index:", segment_index)
        # print("grid_list_v1333:", grid_list_v1)
        for grid1 in grid_list_v1:
            #print("grid1:", grid1)
            # print("candidates[grid1]:", candidates[grid1])
            if not candidates[grid1]:
                # progress_bar.update(1)
                continue
            segment1 = segments1[grid1][segment_index]
            #print("segment1:", segment1)
            for grid2 in candidates[grid1][:]:  # 遍历候选集合中的网格
                #print("grid2:", grid2)
                #print("segments2[grid2]:", segments2[grid2])
                #print("len(segments2[grid2]):", len(segments2[grid2]))
                #print("segment_index:", segment_index)
                if segment_index < len(segments2[grid2]):
                    segment2 = segments2[grid2][segment_index]
                    dist = segment_similarity(segment1, segment2, distance_metric)
                    #print("segment1:", segment1)
                    #print("segment2:", segment2)
                    if first_segment:
                        threshold_check_set.add(dist)
                    elif dist > threshold:
                        mismatch_counts[grid1][grid2] += 1
                        if mismatch_counts[grid1][grid2] > max_mismatches:
                            candidates[grid1].remove(grid2)
                            # tqdm.write(f"Segment {segment_index}, Grid1 {grid1} removed Grid2 {grid2} from candidate set")
                    # 打印中间信息
            #         log_file.write(f"Segment {segment_index}, Grid1 {grid1}, Grid2 {grid2}, Distance: {dist}\n")
            # progress_bar.update(1)
        if first_segment:
            #  设置阈值
            threshold_check_set = sorted(threshold_check_set)
            # print("threshold_check_set:", threshold_check_set)
            # print("threshold_check_set_len:", len(threshold_check_set))
            # print("mismatch_distance_factor:", mismatch_distance_factor)
            threshold = threshold_check_set[len(threshold_check_set) // mismatch_distance_factor]
            # print("threshold:", threshold)
            # threshold = 300
            first_segment = False
    
    # progress_bar.close()
    # log_file.close()

    # # 通过将所有段匹配结果求和构建匹配结果
    # matching_result = {}
    # for grid1 in range(grid_num1):
    #     if candidates[grid1]:
    #         # 选取候选集合中匹配结果最好的网格作为匹配结果
    #         best_match = min(candidates[grid1], key=lambda g: sum(
    #             segment_similarity(seg1, seg2, distance_metric) for seg1, seg2 in zip(segments1[grid1], segments2[g])
    #         ))
    #         matching_result[grid1] = best_match
    #     else:
    #         matching_result[grid1] = None

    # 通过对整段进行另一逻辑匹配构建匹配结果
    matching_result = []
    for grid1 in grid_list_v1:
        best_match = None
        min_distance = float('inf')
        for grid2 in candidates[grid1]:
            dist = segment_similarity(motion_status_matrix1[grid1], motion_status_matrix2[grid2], distance_metric="logic_and")
            if dist < min_distance:
                min_distance = dist
                best_match = grid2
            matching_result.append((grid1, grid2, min_distance))
        # if min_distance < float('inf'):
        #     matching_result.append((grid1, best_match, min_distance))
    # matching_result = sorted(matching_result, key=lambda x: (x[0], x[2]))
    end_time = time.time()
    foldername = "/mnt/mDisk/Project/CityCam/Output/citydata.txt"
    with open(foldername, "a") as f:
        print("Match_匹配耗时：", end_time - start_time, file=f)
    matching_result = sorted(matching_result, key=lambda x: (x[2]))

    return matching_result


def propagate_matching_result(match_result, motion_status_per_grid1, motion_status_per_grid2, num_rows1, num_cols1, num_rows2, num_cols2, parameters, shifting_flag=False):
    """
    将匹配结果传播到整个网格
    Args:
        match_result: 匹配结果列表
        motion_status_per_grid1: 视频1的网格状态序列
        motion_status_per_grid2: 视频2的网格状态序列
        parameters: 参数字典
    Return:
        匹配结果列表
    """
    distance_metric = parameters['distance_metric']
    propagate_step = parameters['propagate_step']

    # 判断网格是否经过propagate
    grid_bit_map = [i for i in range(0,len(motion_status_per_grid1))]

    match_result_propagated = match_result.copy()

    for match in match_result:
        index1, index2, point = match
        x1, y1 = index1 // num_cols1, index1 % num_cols1
        x2, y2 = index2 // num_cols2, index2 % num_cols2
        for i in range(-propagate_step, propagate_step + 1):
            for j in range(-propagate_step, propagate_step + 1):

                min_dist = len(motion_status_per_grid1[0])
                max_dist = 0
                better_match = []

                new_x1, new_y1 = x1 + i, y1 + j
                if 0 <= new_x1 < num_rows1 and 0 <= new_y1 < num_cols1:
                    new_index1 = new_x1 * num_cols1 + new_y1
                    if new_index1 not in grid_bit_map:
                        continue
                    grid_bit_map.remove(new_index1)
                    # 如果网格没有运动元素，不进行传播
                    if motion_status_per_grid1[new_index1].sum() == 0:
                        continue

                    new_x2, new_y2 = x2 + i, y2 + j
                    if 0 <= new_x2 < num_rows2 and 0 <= new_y2 < num_cols2:
                        for k in range(-propagate_step, propagate_step + 1):
                            for l in range(-propagate_step, propagate_step + 1):
                                new_x2, new_y2 = new_x2 + k, new_y2 + l
                                if 0 <= new_x2 < num_rows2 and 0 <= new_y2 < num_cols2:
                                    new_index2 = new_x2 * num_cols2 + new_y2
                                    if motion_status_per_grid2[new_index2].sum() == 0:
                                        continue
                                    dist = segment_similarity(motion_status_per_grid1[new_index1], motion_status_per_grid2[new_index2], distance_metric)
                                    if dist == min_dist:
                                        better_match.append(new_index2)
                                    if dist < min_dist:
                                        min_dist = dist
                                        better_match.clear()
                                        better_match.append(new_index2)
                                    if dist > max_dist:
                                        max_dist = dist
                        # for match_index in better_match:
                        # match_index = better_match[0]
                    
                        ori_match_grid_list = [triplet[1] for triplet in match_result_propagated if triplet[0] == new_index1]
                        
                        # 新的匹配结果在之前的匹配结果中，说明确实应该存在对应的区域
                        if ori_match_grid_list:
                            common_elements = list(filter(lambda x: x in better_match, ori_match_grid_list))
                            # 说明新的匹配结果与之前的匹配结果有重叠
                            if common_elements:
                                for match_index in better_match:
                                    # 加入新匹配结果中在之前匹配结果中没有的元素
                                    if match_index not in common_elements:
                                        match_result_propagated.append((new_index1, match_index, min_dist))
                            
                            # 新的匹配结果与之前的匹配结果没有重叠
                            else:
                                ori_match_grid_point = segment_similarity(motion_status_per_grid1[new_index1], motion_status_per_grid2[ori_match_grid_list[0]], distance_metric)
                                ratio = 0.6
                                if min_dist < ratio * ori_match_grid_point + (1 - ratio) * max_dist:
                                    # 确实需要更新，删除原来的匹配结果，加入新的匹配结果
                                    match_result_propagated = [triplet for triplet in match_result_propagated if triplet[0] != new_index1]
                                    for match_index in better_match:
                                        match_result_propagated.append((new_index1, match_index, min_dist))

                        # 原本match_list没有index1的匹配结果，这时需要加入
                        else:
                            ori_match_grid_point = segment_similarity(motion_status_per_grid1[index1], motion_status_per_grid2[index2], distance_metric)
                            # 不能直接加入，因为此时的new_index1可能是非对应区域的运动网格，需要和之前的匹配结果进行比较以考虑是否加入
                            ratio = 0.6
                            if min_dist < ratio * ori_match_grid_point + (1 - ratio) * max_dist:
                                for match_index in better_match:
                                    match_result_propagated.append((new_index1, match_index, min_dist))
    return match_result_propagated

# # 示例用法
# motion_status_matrix1 = np.random.randint(0, 2, (16, 100))  # 16个网格，每个网格有100个状态
# motion_status_matrix2 = np.random.randint(0, 2, (16, 100))  # 16个网格，每个网格有100个状态
# segment_length = 10
# threshold = 0.2  # Hamming 距离阈值

# matching_result = find_matching_grid_with_segment(motion_status_matrix1, motion_status_matrix2, segment_length, 1)
# print("匹配结果：")
# print(matching_result)
