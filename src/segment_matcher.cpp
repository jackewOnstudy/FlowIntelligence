#include "video_matcher.h"
#include <algorithm>
#include <set>
#include <iostream>
#include <chrono>
#include <fstream>

namespace VideoMatcher {

std::vector<std::vector<bool>> SegmentMatcher::segmentSequence(const std::vector<bool>& sequence, int segment_length) {
    // 对应Python的segment_sequence函数
    std::vector<std::vector<bool>> segments;
    
    for (size_t i = 0; i < sequence.size(); i += segment_length) {
        size_t end = std::min(i + segment_length, sequence.size());
        std::vector<bool> segment(sequence.begin() + i, sequence.begin() + end);
        segments.push_back(segment);
    }
    
    return segments;
}

std::vector<cv::Mat> SegmentMatcher::segmentMatrix(const cv::Mat& matrix, int segment_length) {
    // 对应Python的segment_matrix函数
    std::vector<cv::Mat> segments;
    
    for (int i = 0; i < matrix.cols; i += segment_length) {
        int end = std::min(i + segment_length, matrix.cols);
        cv::Mat segment = matrix.colRange(i, end).clone();
        segments.push_back(segment);
    }
    
    return segments;
}

std::vector<MatchTriplet> SegmentMatcher::findMatchingGridWithSegment(
    const cv::Mat& motion_status_matrix1, const cv::Mat& motion_status_matrix2,
    const Parameters& parameters, const std::map<int, std::set<int>>& sorted_large_grid_corre_small_dict,
    int small_grid_cols, int large_grid_cols, bool shifting_flag) {
    
    // 对应Python的find_matching_grid_with_segment函数
    int grid_num1 = motion_status_matrix1.rows;
    int grid_num2 = motion_status_matrix2.rows;
    int seq_len1 = motion_status_matrix1.cols;
    int seq_len2 = motion_status_matrix2.cols;
    
    if (seq_len1 != seq_len2) {
        throw std::invalid_argument("两个矩阵的网格状态序列长度应相同");
    }
    
    int segment_length = parameters.segment_length;
    int max_mismatches = parameters.max_mismatches;
    std::string distance_metric = parameters.distance_metric;
    int select_grid_factor = parameters.select_grid_factor;
    int mismatch_distance_factor = parameters.mismatch_distance_factor;
    
    std::vector<int> grid_list_v1;
    std::vector<std::vector<int>> candidates(grid_num1);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 初始化候选集合
    if (sorted_large_grid_corre_small_dict.empty()) {
        // 第一次匹配，根据运动元素数量筛选
        int one_threshold = seq_len1 / select_grid_factor;
        
        for (int i = 0; i < grid_num1; ++i) {
            // 计算网格运动元素总数
            int motion_count = 0;
            for (int j = 0; j < seq_len1; ++j) {
                if (motion_status_matrix1.at<uchar>(i, j) > 0) {
                    motion_count++;
                }
            }
            
            if (motion_count >= one_threshold) {
                grid_list_v1.push_back(i);
                
                // 为该网格找候选匹配网格
                std::vector<int> valid_candidates;
                for (int j = 0; j < grid_num2; ++j) {
                    int motion_count_j = 0;
                    for (int k = 0; k < seq_len2; ++k) {
                        if (motion_status_matrix2.at<uchar>(j, k) > 0) {
                            motion_count_j++;
                        }
                    }
                    if (motion_count_j >= one_threshold) {
                        valid_candidates.push_back(j);
                    }
                }
                candidates[i] = valid_candidates;
            }
        }
    } else {
        // 基于上一层级的匹配结果
        max_mismatches = std::max(1, max_mismatches - 1);
        mismatch_distance_factor = std::max(2, mismatch_distance_factor - 2);
        
        for (const auto& pair : sorted_large_grid_corre_small_dict) {
            int key = pair.first;
            const std::set<int>& v2_small_grid_set = pair.second;
            
            // 计算v1对应的小网格集合
            std::set<int> v1_small_grid_set = VideoMatcherUtils::getSmallIndexInLarge(
                key, large_grid_cols, small_grid_cols, 2, 2, shifting_flag);
            
            // 过滤掉没有运动的网格
            std::vector<int> valid_v1_grids;
            for (int grid_idx : v1_small_grid_set) {
                int motion_count = 0;
                for (int j = 0; j < seq_len1; ++j) {
                    if (motion_status_matrix1.at<uchar>(grid_idx, j) > 0) {
                        motion_count++;
                    }
                }
                if (motion_count > 0) {
                    valid_v1_grids.push_back(grid_idx);
                    
                    // 为该网格设置候选
                    std::vector<int> valid_candidates;
                    for (int j : v2_small_grid_set) {
                        int motion_count_j = 0;
                        for (int k = 0; k < seq_len2; ++k) {
                            if (motion_status_matrix2.at<uchar>(j, k) > 0) {
                                motion_count_j++;
                            }
                        }
                        if (motion_count_j > 0) {
                            valid_candidates.push_back(j);
                        }
                    }
                    candidates[grid_idx] = valid_candidates;
                }
            }
            
            grid_list_v1.insert(grid_list_v1.end(), valid_v1_grids.begin(), valid_v1_grids.end());
        }
    }
    
    // 初始化不匹配段次数
    std::vector<std::vector<int>> mismatch_counts(grid_num1, std::vector<int>(grid_num2, 0));
    
    // 将所有网格的状态序列分段
    std::vector<std::vector<std::vector<bool>>> segments1(grid_num1);
    std::vector<std::vector<std::vector<bool>>> segments2(grid_num2);
    
    for (int i = 0; i < grid_num1; ++i) {
        std::vector<bool> sequence(seq_len1);
        for (int j = 0; j < seq_len1; ++j) {
            sequence[j] = motion_status_matrix1.at<uchar>(i, j) > 0;
        }
        segments1[i] = segmentSequence(sequence, segment_length);
    }
    
    for (int i = 0; i < grid_num2; ++i) {
        std::vector<bool> sequence(seq_len2);
        for (int j = 0; j < seq_len2; ++j) {
            sequence[j] = motion_status_matrix2.at<uchar>(i, j) > 0;
        }
        segments2[i] = segmentSequence(sequence, segment_length);
    }
    
    int num_segments = segments1.empty() ? 0 : segments1[0].size();
    
    bool first_segment = true;
    std::set<float> threshold_check_set;
    float threshold = 0.0f;
    
    // 分段匹配处理
    for (int segment_index = 0; segment_index < num_segments; ++segment_index) {
        for (int grid1 : grid_list_v1) {
            if (candidates[grid1].empty()) {
                continue;
            }
            
            if (segment_index >= static_cast<int>(segments1[grid1].size())) {
                continue;
            }
            
            const auto& segment1 = segments1[grid1][segment_index];
            
            for (auto it = candidates[grid1].begin(); it != candidates[grid1].end();) {
                int grid2 = *it;
                
                if (segment_index < static_cast<int>(segments2[grid2].size())) {
                    const auto& segment2 = segments2[grid2][segment_index];
                    float dist = DistanceCalculator::segmentSimilarity(segment1, segment2, distance_metric);
                    
                    if (first_segment) {
                        threshold_check_set.insert(dist);
                        ++it;
                    } else if (dist > threshold) {
                        mismatch_counts[grid1][grid2]++;
                        if (mismatch_counts[grid1][grid2] > max_mismatches) {
                            it = candidates[grid1].erase(it);
                        } else {
                            ++it;
                        }
                    } else {
                        ++it;
                    }
                } else {
                    ++it;
                }
            }
        }
        
        if (first_segment) {
            // 设置阈值
            std::vector<float> sorted_thresholds(threshold_check_set.begin(), threshold_check_set.end());
            if (!sorted_thresholds.empty()) {
                int threshold_index = sorted_thresholds.size() / mismatch_distance_factor;
                threshold = sorted_thresholds[threshold_index];
            }
            first_segment = false;
        }
    }
    
    // 构建最终匹配结果
    std::vector<MatchTriplet> matching_result;
    
    for (int grid1 : grid_list_v1) {
        for (int grid2 : candidates[grid1]) {
            // 计算整个序列的距离
            std::vector<bool> seq1(seq_len1), seq2(seq_len2);
            for (int j = 0; j < seq_len1; ++j) {
                seq1[j] = motion_status_matrix1.at<uchar>(grid1, j) > 0;
            }
            for (int j = 0; j < seq_len2; ++j) {
                seq2[j] = motion_status_matrix2.at<uchar>(grid2, j) > 0;
            }
            
            float dist = DistanceCalculator::segmentSimilarity(seq1, seq2, "logic_and");
            matching_result.emplace_back(grid1, grid2, dist);
        }
    }
    
    // 按距离排序
    std::sort(matching_result.begin(), matching_result.end(), 
              [](const MatchTriplet& a, const MatchTriplet& b) {
                  return a.distance < b.distance;
              });
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // 记录匹配耗时
    std::ofstream log_file("/mnt/mDisk/Project/CityCam/Output/citydata.txt", std::ios::app);
    if (log_file.is_open()) {
        log_file << "Match_匹配耗时: " << duration.count() << " ms\n";
        log_file.close();
    }
    
    return matching_result;
}

std::vector<MatchTriplet> SegmentMatcher::propagateMatchingResult(
    const std::vector<MatchTriplet>& match_result,
    const cv::Mat& motion_status_per_grid1, const cv::Mat& motion_status_per_grid2,
    int num_rows1, int num_cols1, int num_rows2, int num_cols2,
    const Parameters& parameters, bool shifting_flag) {
    
    // 对应Python的propagate_matching_result函数
    std::string distance_metric = parameters.distance_metric;
    int propagate_step = parameters.propagate_step;
    
    // 判断网格是否经过propagate
    std::set<int> grid_bit_map;
    for (int i = 0; i < motion_status_per_grid1.rows; ++i) {
        grid_bit_map.insert(i);
    }
    
    std::vector<MatchTriplet> match_result_propagated = match_result;
    
    for (const auto& match : match_result) {
        int index1 = match.grid1;
        int index2 = match.grid2;
        
        int x1 = index1 / num_cols1;
        int y1 = index1 % num_cols1;
        int x2 = index2 / num_cols2;
        int y2 = index2 % num_cols2;
        
        for (int i = -propagate_step; i <= propagate_step; ++i) {
            for (int j = -propagate_step; j <= propagate_step; ++j) {
                int new_x1 = x1 + i;
                int new_y1 = y1 + j;
                
                if (new_x1 >= 0 && new_x1 < num_rows1 && new_y1 >= 0 && new_y1 < num_cols1) {
                    int new_index1 = new_x1 * num_cols1 + new_y1;
                    
                    if (grid_bit_map.find(new_index1) == grid_bit_map.end()) {
                        continue;
                    }
                    grid_bit_map.erase(new_index1);
                    
                    // 检查网格是否有运动元素
                    int motion_sum1 = 0;
                    for (int col = 0; col < motion_status_per_grid1.cols; ++col) {
                        if (motion_status_per_grid1.at<uchar>(new_index1, col) > 0) {
                            motion_sum1++;
                        }
                    }
                    if (motion_sum1 == 0) {
                        continue;
                    }
                    
                    float min_dist = static_cast<float>(motion_status_per_grid1.cols);
                    float max_dist = 0.0f;
                    std::vector<int> better_match;
                    
                    int new_x2 = x2 + i;
                    int new_y2 = y2 + j;
                    
                    if (new_x2 >= 0 && new_x2 < num_rows2 && new_y2 >= 0 && new_y2 < num_cols2) {
                        for (int k = -propagate_step; k <= propagate_step; ++k) {
                            for (int l = -propagate_step; l <= propagate_step; ++l) {
                                int final_x2 = new_x2 + k;
                                int final_y2 = new_y2 + l;
                                
                                if (final_x2 >= 0 && final_x2 < num_rows2 && 
                                    final_y2 >= 0 && final_y2 < num_cols2) {
                                    
                                    int new_index2 = final_x2 * num_cols2 + final_y2;
                                    
                                    // 检查网格是否有运动元素
                                    int motion_sum2 = 0;
                                    for (int col = 0; col < motion_status_per_grid2.cols; ++col) {
                                        if (motion_status_per_grid2.at<uchar>(new_index2, col) > 0) {
                                            motion_sum2++;
                                        }
                                    }
                                    if (motion_sum2 == 0) {
                                        continue;
                                    }
                                    
                                    // 计算距离
                                    std::vector<bool> seq1(motion_status_per_grid1.cols);
                                    std::vector<bool> seq2(motion_status_per_grid2.cols);
                                    
                                    for (int col = 0; col < motion_status_per_grid1.cols; ++col) {
                                        seq1[col] = motion_status_per_grid1.at<uchar>(new_index1, col) > 0;
                                    }
                                    for (int col = 0; col < motion_status_per_grid2.cols; ++col) {
                                        seq2[col] = motion_status_per_grid2.at<uchar>(new_index2, col) > 0;
                                    }
                                    
                                    float dist = DistanceCalculator::segmentSimilarity(seq1, seq2, distance_metric);
                                    
                                    if (std::abs(dist - min_dist) < 1e-6) {
                                        better_match.push_back(new_index2);
                                    } else if (dist < min_dist) {
                                        min_dist = dist;
                                        better_match.clear();
                                        better_match.push_back(new_index2);
                                    }
                                    
                                    if (dist > max_dist) {
                                        max_dist = dist;
                                    }
                                }
                            }
                        }
                        
                        // 检查是否存在原有匹配
                        std::vector<int> ori_match_grid_list;
                        for (const auto& triplet : match_result_propagated) {
                            if (triplet.grid1 == new_index1) {
                                ori_match_grid_list.push_back(triplet.grid2);
                            }
                        }
                        
                        if (!ori_match_grid_list.empty()) {
                            // 检查重叠
                            std::vector<int> common_elements;
                            for (int match_idx : better_match) {
                                if (std::find(ori_match_grid_list.begin(), ori_match_grid_list.end(), match_idx) 
                                    != ori_match_grid_list.end()) {
                                    common_elements.push_back(match_idx);
                                }
                            }
                            
                            if (!common_elements.empty()) {
                                // 有重叠，添加新的匹配
                                for (int match_idx : better_match) {
                                    if (std::find(common_elements.begin(), common_elements.end(), match_idx) 
                                        == common_elements.end()) {
                                        match_result_propagated.emplace_back(new_index1, match_idx, min_dist);
                                    }
                                }
                            } else {
                                // 没有重叠，比较质量
                                std::vector<bool> seq1_ori(motion_status_per_grid1.cols);
                                std::vector<bool> seq2_ori(motion_status_per_grid2.cols);
                                
                                for (int col = 0; col < motion_status_per_grid1.cols; ++col) {
                                    seq1_ori[col] = motion_status_per_grid1.at<uchar>(new_index1, col) > 0;
                                }
                                for (int col = 0; col < motion_status_per_grid2.cols; ++col) {
                                    seq2_ori[col] = motion_status_per_grid2.at<uchar>(ori_match_grid_list[0], col) > 0;
                                }
                                
                                float ori_match_grid_point = DistanceCalculator::segmentSimilarity(
                                    seq1_ori, seq2_ori, distance_metric);
                                
                                float ratio = 0.6f;
                                if (min_dist < ratio * ori_match_grid_point + (1.0f - ratio) * max_dist) {
                                    // 更新匹配结果
                                    auto it = std::remove_if(match_result_propagated.begin(), match_result_propagated.end(),
                                        [new_index1](const MatchTriplet& triplet) {
                                            return triplet.grid1 == new_index1;
                                        });
                                    match_result_propagated.erase(it, match_result_propagated.end());
                                    
                                    for (int match_idx : better_match) {
                                        match_result_propagated.emplace_back(new_index1, match_idx, min_dist);
                                    }
                                }
                            }
                        } else {
                            // 原本没有匹配结果，比较后决定是否添加
                            std::vector<bool> seq1_ref(motion_status_per_grid1.cols);
                            std::vector<bool> seq2_ref(motion_status_per_grid2.cols);
                            
                            for (int col = 0; col < motion_status_per_grid1.cols; ++col) {
                                seq1_ref[col] = motion_status_per_grid1.at<uchar>(index1, col) > 0;
                            }
                            for (int col = 0; col < motion_status_per_grid2.cols; ++col) {
                                seq2_ref[col] = motion_status_per_grid2.at<uchar>(index2, col) > 0;
                            }
                            
                            float ori_match_grid_point = DistanceCalculator::segmentSimilarity(
                                seq1_ref, seq2_ref, distance_metric);
                            
                            float ratio = 0.6f;
                            if (min_dist < ratio * ori_match_grid_point + (1.0f - ratio) * max_dist) {
                                for (int match_idx : better_match) {
                                    match_result_propagated.emplace_back(new_index1, match_idx, min_dist);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return match_result_propagated;
}

} // namespace VideoMatcher 