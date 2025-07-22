#include "video_matcher.h"
#include <algorithm>
#include <set>
#include <unordered_set>  // 新增：高效集合操作
#include <unordered_map>  // 新增：高效映射操作
#include <iostream>
#include <chrono>
#include <fstream>
#include <mutex>  // 新增：互斥锁支持
#ifdef _OPENMP
#include <omp.h>  // 新增：OpenMP支持
#endif

namespace VideoMatcher {

std::vector<std::vector<bool>> SegmentMatcher::segmentSequence(const std::vector<bool>& sequence, int segment_length) {
    // 对应Python的segment_sequence函数 - 优化版本
    if (sequence.empty() || segment_length <= 0) {
        return {};
    }
    
    std::vector<std::vector<bool>> segments;
    size_t num_segments = (sequence.size() + segment_length - 1) / segment_length;
    segments.reserve(num_segments);  // 预分配内存
    
    for (size_t i = 0; i < sequence.size(); i += segment_length) {
        size_t end = std::min(i + segment_length, sequence.size());
        segments.emplace_back(sequence.begin() + i, sequence.begin() + end);
    }
    
    return segments;
}

std::vector<cv::Mat> SegmentMatcher::segmentMatrix(const cv::Mat& matrix, int segment_length) {
    // 对应Python的segment_matrix函数 - 优化版本
    if (matrix.empty() || segment_length <= 0) {
        return {};
    }
    
    std::vector<cv::Mat> segments;
    size_t num_segments = (matrix.cols + segment_length - 1) / segment_length;
    segments.reserve(num_segments);  // 预分配内存
    
    for (int i = 0; i < matrix.cols; i += segment_length) {
        int end = std::min(i + segment_length, matrix.cols);
        segments.push_back(matrix.colRange(i, end).clone());
    }
    
    return segments;
}

std::vector<MatchTriplet> SegmentMatcher::findMatchingGridWithSegment(
    const cv::Mat& motion_status_matrix1, const cv::Mat& motion_status_matrix2,
    const Parameters& parameters, const std::map<int, std::set<int>>& sorted_large_grid_corre_small_dict,
    int small_grid_cols, int large_grid_cols, bool shifting_flag) {
    
    // 对应Python的find_matching_grid_with_segment函数 - 优化版本
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
    std::vector<std::unordered_set<int>> candidates(grid_num1);  // 使用unordered_set优化查找
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 初始化候选集合
    if (sorted_large_grid_corre_small_dict.empty()) {
        // 第一次匹配，根据运动元素数量筛选 - 并行化优化
        int one_threshold = seq_len1 / select_grid_factor;
        
        // 预计算运动计数 - 新增优化
        std::vector<int> motion_counts1(grid_num1), motion_counts2(grid_num2);
        
        #pragma omp parallel for schedule(static) if(grid_num1 > 100)
        for (int i = 0; i < grid_num1; ++i) {
            int motion_count = 0;
            for (int j = 0; j < seq_len1; ++j) {
                if (motion_status_matrix1.at<uchar>(i, j) > 0) {
                    motion_count++;
                }
            }
            motion_counts1[i] = motion_count;
        }
        
        #pragma omp parallel for schedule(static) if(grid_num2 > 100)
        for (int i = 0; i < grid_num2; ++i) {
            int motion_count = 0;
            for (int j = 0; j < seq_len2; ++j) {
                if (motion_status_matrix2.at<uchar>(i, j) > 0) {
                    motion_count++;
                }
            }
            motion_counts2[i] = motion_count;
        }
        
        // 构建候选列表
        for (int i = 0; i < grid_num1; ++i) {
            if (motion_counts1[i] >= one_threshold) {
                grid_list_v1.push_back(i);
                
                // 为该网格找候选匹配网格
                std::unordered_set<int> valid_candidates;
                for (int j = 0; j < grid_num2; ++j) {
                    if (motion_counts2[j] >= one_threshold) {
                        valid_candidates.insert(j);
                    }
                }
                candidates[i] = std::move(valid_candidates);
            }
        }
    } else {
        // 基于上一层级的匹配结果 - 优化版本
        max_mismatches = std::max(1, max_mismatches - 1);
        mismatch_distance_factor = std::max(2, mismatch_distance_factor - 2);
        
        // 预计算运动计数以避免重复计算
        std::vector<int> motion_counts1(grid_num1), motion_counts2(grid_num2);
        
        #pragma omp parallel for schedule(static) if(grid_num1 > 100)
        for (int i = 0; i < grid_num1; ++i) {
            int motion_count = 0;
            for (int j = 0; j < seq_len1; ++j) {
                if (motion_status_matrix1.at<uchar>(i, j) > 0) {
                    motion_count++;
                }
            }
            motion_counts1[i] = motion_count;
        }
        
        #pragma omp parallel for schedule(static) if(grid_num2 > 100)
        for (int i = 0; i < grid_num2; ++i) {
            int motion_count = 0;
            for (int j = 0; j < seq_len2; ++j) {
                if (motion_status_matrix2.at<uchar>(i, j) > 0) {
                    motion_count++;
                }
            }
            motion_counts2[i] = motion_count;
        }
        
        for (const auto& pair : sorted_large_grid_corre_small_dict) {
            int key = pair.first;
            const std::set<int>& v2_small_grid_set = pair.second;
            
            // 计算v1对应的小网格集合
            std::set<int> v1_small_grid_set = VideoMatcherUtils::getSmallIndexInLarge(
                key, large_grid_cols, small_grid_cols, 2, 2, shifting_flag);
            
            // 过滤掉没有运动的网格
            std::vector<int> valid_v1_grids;
            for (int grid_idx : v1_small_grid_set) {
                if (grid_idx < grid_num1 && motion_counts1[grid_idx] > 0) {
                    valid_v1_grids.push_back(grid_idx);
                    
                    // 为该网格设置候选
                    std::unordered_set<int> valid_candidates;
                    for (int j : v2_small_grid_set) {
                        if (j < grid_num2 && motion_counts2[j] > 0) {
                            valid_candidates.insert(j);
                        }
                    }
                    candidates[grid_idx] = std::move(valid_candidates);
                }
            }
            
            grid_list_v1.insert(grid_list_v1.end(), valid_v1_grids.begin(), valid_v1_grids.end());
        }
    }
    
    // 初始化不匹配段次数 - 使用稀疏矩阵优化
    std::unordered_map<int, std::unordered_map<int, int>> mismatch_counts;
    
    // 预分配分段数据 - 新增优化
    std::vector<std::vector<std::vector<bool>>> segments1, segments2;
    segments1.resize(grid_num1);
    segments2.resize(grid_num2);
    
    // 并行化分段预处理
    #pragma omp parallel for schedule(dynamic) if(grid_num1 > 50)
    for (int i = 0; i < grid_num1; ++i) {
        if (std::find(grid_list_v1.begin(), grid_list_v1.end(), i) != grid_list_v1.end()) {
            std::vector<bool> sequence(seq_len1);
            for (int j = 0; j < seq_len1; ++j) {
                sequence[j] = motion_status_matrix1.at<uchar>(i, j) > 0;
            }
            segments1[i] = segmentSequence(sequence, segment_length);
        }
    }
    
    #pragma omp parallel for schedule(dynamic) if(grid_num2 > 50)
    for (int i = 0; i < grid_num2; ++i) {
        std::vector<bool> sequence(seq_len2);
        for (int j = 0; j < seq_len2; ++j) {
            sequence[j] = motion_status_matrix2.at<uchar>(i, j) > 0;
        }
        segments2[i] = segmentSequence(sequence, segment_length);
    }
    
    int num_segments = segments1.empty() || segments1[0].empty() ? 0 : segments1[0].size();
    
    bool first_segment = true;
    std::set<float> threshold_check_set;
    float threshold = 0.0f;
    
    // 分段匹配处理 - 优化版本
    for (int segment_index = 0; segment_index < num_segments; ++segment_index) {
        // 使用OpenMP并行化外层循环
        #pragma omp parallel for schedule(dynamic) if(grid_list_v1.size() > 10)
        for (size_t idx = 0; idx < grid_list_v1.size(); ++idx) {
            int grid1 = grid_list_v1[idx];
            
            if (candidates[grid1].empty()) {
                continue;
            }
            
            if (segment_index >= static_cast<int>(segments1[grid1].size())) {
                continue;
            }
            
            const auto& segment1 = segments1[grid1][segment_index];
            std::vector<int> to_remove;  // 本地收集要移除的候选
            
            for (auto it = candidates[grid1].begin(); it != candidates[grid1].end(); ++it) {
                int grid2 = *it;
                
                if (segment_index < static_cast<int>(segments2[grid2].size())) {
                    const auto& segment2 = segments2[grid2][segment_index];
                    float dist = DistanceCalculator::segmentSimilarity(segment1, segment2, distance_metric);
                    
                    if (first_segment) {
                        #pragma omp critical
                        threshold_check_set.insert(dist);
                    } else if (dist > threshold) {
                        #pragma omp critical
                        {
                            mismatch_counts[grid1][grid2]++;
                            if (mismatch_counts[grid1][grid2] > max_mismatches) {
                                to_remove.push_back(grid2);
                            }
                        }
                    }
                }
            }
            
            // 移除超过阈值的候选
            if (!to_remove.empty()) {
                #pragma omp critical
                for (int grid2 : to_remove) {
                    candidates[grid1].erase(grid2);
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
    
    // 构建最终匹配结果 - 并行化优化
    std::vector<MatchTriplet> matching_result;
    std::vector<std::vector<MatchTriplet>> thread_results;
    
    #pragma omp parallel
    {
        std::vector<MatchTriplet> local_result;
        
        #pragma omp for schedule(dynamic)
        for (size_t idx = 0; idx < grid_list_v1.size(); ++idx) {
            int grid1 = grid_list_v1[idx];
            
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
                local_result.emplace_back(grid1, grid2, dist);
            }
        }
        
        #pragma omp critical
        thread_results.push_back(std::move(local_result));
    }
    
    // 合并线程结果
    for (const auto& thread_result : thread_results) {
        matching_result.insert(matching_result.end(), thread_result.begin(), thread_result.end());
    }
    
    // 按距离排序
    std::sort(matching_result.begin(), matching_result.end(), 
              [](const MatchTriplet& a, const MatchTriplet& b) {
                  return a.distance < b.distance;
              });
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // 记录匹配耗时
    std::ofstream log_file("./Output/citydata.txt", std::ios::app);
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
    
    // 对应Python的propagate_matching_result函数 - 优化版本
    (void)shifting_flag; // 抑制未使用参数警告
    
    if (match_result.empty()) {
        return match_result;
    }
    
    std::string distance_metric = parameters.distance_metric;
    int propagate_step = parameters.propagate_step;
    
    // 使用unordered_set提高查找效率 - 新增优化
    std::unordered_set<int> grid_bit_map;
    for (int i = 0; i < motion_status_per_grid1.rows; ++i) {
        grid_bit_map.insert(i);
    }
    
    std::vector<MatchTriplet> match_result_propagated = match_result;
    
    // 预计算运动计数以避免重复计算 - 新增优化
    std::vector<int> motion_counts1(motion_status_per_grid1.rows);
    std::vector<int> motion_counts2(motion_status_per_grid2.rows);
    
    #pragma omp parallel for schedule(static) if(motion_status_per_grid1.rows > 100)
    for (int i = 0; i < motion_status_per_grid1.rows; ++i) {
        int motion_count = 0;
        for (int col = 0; col < motion_status_per_grid1.cols; ++col) {
            if (motion_status_per_grid1.at<uchar>(i, col) > 0) {
                motion_count++;
            }
        }
        motion_counts1[i] = motion_count;
    }
    
    #pragma omp parallel for schedule(static) if(motion_status_per_grid2.rows > 100)
    for (int i = 0; i < motion_status_per_grid2.rows; ++i) {
        int motion_count = 0;
        for (int col = 0; col < motion_status_per_grid2.cols; ++col) {
            if (motion_status_per_grid2.at<uchar>(i, col) > 0) {
                motion_count++;
            }
        }
        motion_counts2[i] = motion_count;
    }
    
    // 并行化处理匹配结果 - 新增优化
    std::vector<std::mutex> grid_mutexes(motion_status_per_grid1.rows);  // 每个网格一个互斥锁
    std::vector<std::vector<MatchTriplet>> thread_additions;
    
    #pragma omp parallel
    {
        std::vector<MatchTriplet> local_additions;
        std::unordered_set<int> local_processed;  // 线程本地的已处理网格
        
        #pragma omp for schedule(dynamic)
        for (size_t match_idx = 0; match_idx < match_result.size(); ++match_idx) {
            const auto& match = match_result[match_idx];
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
                        
                        // 检查是否已经被处理过
                        bool should_process = false;
                        #pragma omp critical(grid_check)
                        {
                            if (grid_bit_map.find(new_index1) != grid_bit_map.end()) {
                                grid_bit_map.erase(new_index1);
                                should_process = true;
                            }
                        }
                        
                        if (!should_process || motion_counts1[new_index1] == 0) {
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
                                        
                                        if (motion_counts2[new_index2] == 0) {
                                            continue;
                                        }
                                        
                                        // 预分配序列以减少内存分配 - 新增优化
                                        std::vector<bool> seq1, seq2;
                                        seq1.reserve(motion_status_per_grid1.cols);
                                        seq2.reserve(motion_status_per_grid2.cols);
                                        
                                        for (int col = 0; col < motion_status_per_grid1.cols; ++col) {
                                            seq1.push_back(motion_status_per_grid1.at<uchar>(new_index1, col) > 0);
                                        }
                                        for (int col = 0; col < motion_status_per_grid2.cols; ++col) {
                                            seq2.push_back(motion_status_per_grid2.at<uchar>(new_index2, col) > 0);
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
                            
                            // 检查是否存在原有匹配 - 优化版本
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
                                            local_additions.emplace_back(new_index1, match_idx, min_dist);
                                        }
                                    }
                                } else {
                                    // 没有重叠，比较质量
                                    std::vector<bool> seq1_ori, seq2_ori;
                                    seq1_ori.reserve(motion_status_per_grid1.cols);
                                    seq2_ori.reserve(motion_status_per_grid2.cols);
                                    
                                    for (int col = 0; col < motion_status_per_grid1.cols; ++col) {
                                        seq1_ori.push_back(motion_status_per_grid1.at<uchar>(new_index1, col) > 0);
                                    }
                                    for (int col = 0; col < motion_status_per_grid2.cols; ++col) {
                                        seq2_ori.push_back(motion_status_per_grid2.at<uchar>(ori_match_grid_list[0], col) > 0);
                                    }
                                    
                                    float ori_match_grid_point = DistanceCalculator::segmentSimilarity(
                                        seq1_ori, seq2_ori, distance_metric);
                                    
                                    float ratio = 0.6f;
                                    if (min_dist < ratio * ori_match_grid_point + (1.0f - ratio) * max_dist) {
                                        // 标记需要更新匹配结果（在critical section中处理）
                                        for (int match_idx : better_match) {
                                            local_additions.emplace_back(new_index1, match_idx, min_dist);
                                        }
                                    }
                                }
                            } else {
                                // 原本没有匹配结果，比较后决定是否添加
                                std::vector<bool> seq1_ref, seq2_ref;
                                seq1_ref.reserve(motion_status_per_grid1.cols);
                                seq2_ref.reserve(motion_status_per_grid2.cols);
                                
                                for (int col = 0; col < motion_status_per_grid1.cols; ++col) {
                                    seq1_ref.push_back(motion_status_per_grid1.at<uchar>(index1, col) > 0);
                                }
                                for (int col = 0; col < motion_status_per_grid2.cols; ++col) {
                                    seq2_ref.push_back(motion_status_per_grid2.at<uchar>(index2, col) > 0);
                                }
                                
                                float ori_match_grid_point = DistanceCalculator::segmentSimilarity(
                                    seq1_ref, seq2_ref, distance_metric);
                                
                                float ratio = 0.6f;
                                if (min_dist < ratio * ori_match_grid_point + (1.0f - ratio) * max_dist) {
                                    for (int match_idx : better_match) {
                                        local_additions.emplace_back(new_index1, match_idx, min_dist);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        #pragma omp critical
        thread_additions.push_back(std::move(local_additions));
    }
    
    // 合并所有线程的添加结果
    for (const auto& additions : thread_additions) {
        for (const auto& triplet : additions) {
            // 移除现有的匹配（如果有）
            auto it = std::remove_if(match_result_propagated.begin(), match_result_propagated.end(),
                [triplet](const MatchTriplet& existing) {
                    return existing.grid1 == triplet.grid1;
                });
            match_result_propagated.erase(it, match_result_propagated.end());
            
            // 添加新的匹配
            match_result_propagated.push_back(triplet);
        }
    }
    
    return match_result_propagated;
}

} // namespace VideoMatcher 