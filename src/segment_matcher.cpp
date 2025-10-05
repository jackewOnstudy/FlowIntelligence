#include "video_matcher.h"
#include <algorithm>
#include <set>
#include <unordered_set>  
#include <unordered_map> 
#include <iostream>
#include <chrono>
#include <fstream>
#include <mutex>  
#include <iomanip>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>  
#endif

namespace VideoMatcher {
 
static std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%dT%H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

std::vector<std::vector<bool>> SegmentMatcher::segmentSequence(const std::vector<bool>& sequence, int segment_length) {
    if (sequence.empty() || segment_length <= 0) {
        return {};
    }
    
    std::vector<std::vector<bool>> segments;
    size_t num_segments = (sequence.size() + segment_length - 1) / segment_length;
    segments.reserve(num_segments);  
    
    for (size_t i = 0; i < sequence.size(); i += segment_length) {
        size_t end = std::min(i + segment_length, sequence.size());
        segments.emplace_back(sequence.begin() + i, sequence.begin() + end);
    }
    
    return segments;
}

std::vector<cv::Mat> SegmentMatcher::segmentMatrix(const cv::Mat& matrix, int segment_length) {
    if (matrix.empty() || segment_length <= 0) {
        return {};
    }
    
    std::vector<cv::Mat> segments;
    size_t num_segments = (matrix.cols + segment_length - 1) / segment_length;
    segments.reserve(num_segments); 
    
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
    std::vector<std::unordered_set<int>> candidates(grid_num1); 
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    if (sorted_large_grid_corre_small_dict.empty()) {
        int one_threshold = seq_len1 / select_grid_factor;
        
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
        
        for (int i = 0; i < grid_num1; ++i) {
            if (motion_counts1[i] >= one_threshold) {
                grid_list_v1.push_back(i);
                
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
        max_mismatches = std::max(1, max_mismatches - 1);
        mismatch_distance_factor = std::max(2, mismatch_distance_factor - 2);
        
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
            
            std::set<int> v1_small_grid_set = VideoMatcherUtils::getSmallIndexInLarge(
                key, large_grid_cols, small_grid_cols, 2, 2, shifting_flag);
            
            std::vector<int> valid_v1_grids;
            for (int grid_idx : v1_small_grid_set) {
                if (grid_idx < grid_num1 && motion_counts1[grid_idx] > 0) {
                    valid_v1_grids.push_back(grid_idx);
                    
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
    
    std::unordered_map<int, std::unordered_map<int, int>> mismatch_counts;
    
    std::vector<std::vector<std::vector<bool>>> segments1, segments2;
    segments1.resize(grid_num1);
    segments2.resize(grid_num2);
    
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
    
    for (int segment_index = 0; segment_index < num_segments; ++segment_index) {
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
            std::vector<int> to_remove; 
            
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
            
            if (!to_remove.empty()) {
                #pragma omp critical
                for (int grid2 : to_remove) {
                    candidates[grid1].erase(grid2);
                }
            }
        }
        
        if (first_segment) {
            std::vector<float> sorted_thresholds(threshold_check_set.begin(), threshold_check_set.end());
            if (!sorted_thresholds.empty()) {
                int threshold_index = sorted_thresholds.size() / mismatch_distance_factor;
                threshold = sorted_thresholds[threshold_index];
            }
            first_segment = false;
        }
    }
    
    std::vector<MatchTriplet> matching_result;
    std::vector<std::vector<MatchTriplet>> thread_results;
    
    #pragma omp parallel
    {
        std::vector<MatchTriplet> local_result;
        
        #pragma omp for schedule(dynamic)
        for (size_t idx = 0; idx < grid_list_v1.size(); ++idx) {
            int grid1 = grid_list_v1[idx];
            
            for (int grid2 : candidates[grid1]) {
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
    
    for (const auto& thread_result : thread_results) {
        matching_result.insert(matching_result.end(), thread_result.begin(), thread_result.end());
    }
    
    std::sort(matching_result.begin(), matching_result.end(), 
              [](const MatchTriplet& a, const MatchTriplet& b) {
                  return a.distance < b.distance;
              });
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (!parameters.csv_log_file_path.empty()) {
        std::ofstream csv_file(parameters.csv_log_file_path, std::ios::app);
        if (csv_file.is_open()) {
            csv_file << getCurrentTimestamp() << ","
                     << parameters.video_name1 << "," << parameters.video_name2 << ","
                     << "segment_match,,,,,"
                     << duration.count() << "," << matching_result.size() << std::endl;
            csv_file.close();
        }
    }
    
    return matching_result;
}

std::vector<MatchTriplet> SegmentMatcher::propagateMatchingResult(
    const std::vector<MatchTriplet>& match_result,
    const cv::Mat& motion_status_per_grid1, const cv::Mat& motion_status_per_grid2,
    int num_rows1, int num_cols1, int num_rows2, int num_cols2,
    const Parameters& parameters, bool shifting_flag) {
    
    (void)shifting_flag;
    
    if (match_result.empty()) {
        return match_result;
    }
    
    // 辅助数据结构
    struct PropagationContext {
        const std::string& distance_metric;
        int propagate_step;
        int num_rows1, num_cols1, num_rows2, num_cols2;
        const cv::Mat& motion_status_per_grid1;
        const cv::Mat& motion_status_per_grid2;
        std::vector<int> motion_counts1, motion_counts2;
        std::unordered_set<int> grid_bit_map;
        
        PropagationContext(const Parameters& params, int nr1, int nc1, int nr2, int nc2,
                          const cv::Mat& ms1, const cv::Mat& ms2) 
            : distance_metric(params.distance_metric), propagate_step(params.propagate_step),
              num_rows1(nr1), num_cols1(nc1), num_rows2(nr2), num_cols2(nc2),
              motion_status_per_grid1(ms1), motion_status_per_grid2(ms2) {
            
            // 初始化网格位图
            for (int i = 0; i < motion_status_per_grid1.rows; ++i) {
                grid_bit_map.insert(i);
            }
            
            // 预计算运动计数
            motion_counts1.resize(motion_status_per_grid1.rows);
            motion_counts2.resize(motion_status_per_grid2.rows);
            
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
        }
        
        // 生成网格序列
        std::vector<bool> getGridSequence(const cv::Mat& motion_status, int grid_index) const {
            std::vector<bool> sequence;
            sequence.reserve(motion_status.cols);
            for (int col = 0; col < motion_status.cols; ++col) {
                sequence.push_back(motion_status.at<uchar>(grid_index, col) > 0);
            }
            return sequence;
        }
        
        // 坐标转换
        std::pair<int, int> indexToCoord(int index, int num_cols) const {
            return {index / num_cols, index % num_cols};
        }
        
        int coordToIndex(int x, int y, int num_cols) const {
            return x * num_cols + y;
        }
        
        bool isValidCoord(int x, int y, int num_rows, int num_cols) const {
            return x >= 0 && x < num_rows && y >= 0 && y < num_cols;
        }
    };
    
    PropagationContext ctx(parameters, num_rows1, num_cols1, num_rows2, num_cols2,
                          motion_status_per_grid1, motion_status_per_grid2);
    
    std::vector<MatchTriplet> match_result_propagated = match_result;
    
    // 处理单个传播候选的函数
    auto processPropagationCandidate = [&](const MatchTriplet& match, 
                                          std::vector<MatchTriplet>& local_additions) {
        auto [x1, y1] = ctx.indexToCoord(match.grid1, ctx.num_cols1);
        auto [x2, y2] = ctx.indexToCoord(match.grid2, ctx.num_cols2);
        
        // 生成邻域偏移量
        std::vector<std::pair<int, int>> offsets;
        for (int i = -ctx.propagate_step; i <= ctx.propagate_step; ++i) {
            for (int j = -ctx.propagate_step; j <= ctx.propagate_step; ++j) {
                offsets.emplace_back(i, j);
            }
        }
        
        for (const auto& [dx, dy] : offsets) {
            int new_x1 = x1 + dx, new_y1 = y1 + dy;
            if (!ctx.isValidCoord(new_x1, new_y1, ctx.num_rows1, ctx.num_cols1)) continue;
            
            int new_index1 = ctx.coordToIndex(new_x1, new_y1, ctx.num_cols1);
            
            bool should_process = false;
            #pragma omp critical(grid_check)
            {
                if (ctx.grid_bit_map.find(new_index1) != ctx.grid_bit_map.end()) {
                    ctx.grid_bit_map.erase(new_index1);
                    should_process = true;
                }
            }
            
            if (!should_process || ctx.motion_counts1[new_index1] == 0) continue;
            
            // 查找最佳匹配
            float min_dist = static_cast<float>(ctx.motion_status_per_grid1.cols);
            float max_dist = 0.0f;
            std::vector<int> better_match;
            
            int search_x2 = x2 + dx, search_y2 = y2 + dy;
            if (ctx.isValidCoord(search_x2, search_y2, ctx.num_rows2, ctx.num_cols2)) {
                auto seq1 = ctx.getGridSequence(ctx.motion_status_per_grid1, new_index1);
                
                for (const auto& [dx2, dy2] : offsets) {
                    int final_x2 = search_x2 + dx2, final_y2 = search_y2 + dy2;
                    if (!ctx.isValidCoord(final_x2, final_y2, ctx.num_rows2, ctx.num_cols2)) continue;
                    
                    int new_index2 = ctx.coordToIndex(final_x2, final_y2, ctx.num_cols2);
                    if (ctx.motion_counts2[new_index2] == 0) continue;
                    
                    auto seq2 = ctx.getGridSequence(ctx.motion_status_per_grid2, new_index2);
                    float dist = DistanceCalculator::segmentSimilarity(seq1, seq2, ctx.distance_metric);
                    
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
                
                // 判断是否添加新匹配
                std::vector<int> ori_match_grid_list;
                for (const auto& triplet : match_result_propagated) {
                    if (triplet.grid1 == new_index1) {
                        ori_match_grid_list.push_back(triplet.grid2);
                    }
                }
                
                if (!ori_match_grid_list.empty()) {
                    std::vector<int> common_elements;
                    for (int match_idx : better_match) {
                        if (std::find(ori_match_grid_list.begin(), ori_match_grid_list.end(), match_idx) 
                            != ori_match_grid_list.end()) {
                            common_elements.push_back(match_idx);
                        }
                    }
                    
                    if (!common_elements.empty()) {
                        for (int match_idx : better_match) {
                            if (std::find(common_elements.begin(), common_elements.end(), match_idx) 
                                == common_elements.end()) {
                                local_additions.emplace_back(new_index1, match_idx, min_dist);
                            }
                        }
                    } else {
                        auto seq1_ori = ctx.getGridSequence(ctx.motion_status_per_grid1, new_index1);
                        auto seq2_ori = ctx.getGridSequence(ctx.motion_status_per_grid2, ori_match_grid_list[0]);
                        
                        float ori_match_grid_point = DistanceCalculator::segmentSimilarity(
                            seq1_ori, seq2_ori, ctx.distance_metric);
                        
                        float ratio = 0.6f;
                        if (min_dist < ratio * ori_match_grid_point + (1.0f - ratio) * max_dist) {
                            for (int match_idx : better_match) {
                                local_additions.emplace_back(new_index1, match_idx, min_dist);
                            }
                        }
                    }
                } else {
                    auto seq1_ref = ctx.getGridSequence(ctx.motion_status_per_grid1, match.grid1);
                    auto seq2_ref = ctx.getGridSequence(ctx.motion_status_per_grid2, match.grid2);
                    
                    float ori_match_grid_point = DistanceCalculator::segmentSimilarity(
                        seq1_ref, seq2_ref, ctx.distance_metric);
                    
                    float ratio = 0.6f;
                    if (min_dist < ratio * ori_match_grid_point + (1.0f - ratio) * max_dist) {
                        for (int match_idx : better_match) {
                            local_additions.emplace_back(new_index1, match_idx, min_dist);
                        }
                    }
                }
            }
        }
    };
    
    // 并行处理传播
    std::vector<std::vector<MatchTriplet>> thread_additions;
    
    #pragma omp parallel
    {
        std::vector<MatchTriplet> local_additions;
        
        #pragma omp for schedule(dynamic)
        for (size_t match_idx = 0; match_idx < match_result.size(); ++match_idx) {
            processPropagationCandidate(match_result[match_idx], local_additions);
        }
        
        #pragma omp critical
        thread_additions.push_back(std::move(local_additions));
    }
    
    for (const auto& additions : thread_additions) {
        for (const auto& triplet : additions) {
            auto it = std::remove_if(match_result_propagated.begin(), match_result_propagated.end(),
                [triplet](const MatchTriplet& existing) {
                    return existing.grid1 == triplet.grid1;
                });
            match_result_propagated.erase(it, match_result_propagated.end());
            
            match_result_propagated.push_back(triplet);
        }
    }
    
    return match_result_propagated;
}

} // namespace VideoMatcher 