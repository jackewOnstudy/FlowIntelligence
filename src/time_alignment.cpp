#include "time_alignment.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <iostream>

namespace VideoMatcher {

TimeAlignmentResult TimeAlignmentEngine::detectTimeOffset(const cv::Mat& motion_status1, 
                                                          const cv::Mat& motion_status2) {
    TimeAlignmentResult result;
    
    if (motion_status1.rows == 0 || motion_status2.rows == 0 || 
        motion_status1.cols != motion_status2.cols) {
        return result;  // 返回无效结果
    }
    
    // 粗检测：先在降采样的序列上快速检测
    cv::Mat coarse_status1 = motion_status1, coarse_status2 = motion_status2;
    if (params_.use_coarse_detection && motion_status1.cols > 100) {
        coarse_status1 = temporalDownsample(motion_status1, params_.coarse_downsample_factor);
        coarse_status2 = temporalDownsample(motion_status2, params_.coarse_downsample_factor);
    }
    
    // 获取网格维度信息
    GridDimensions grid_dims1(coarse_status1);
    GridDimensions grid_dims2(coarse_status2);
    
    // 创建空间区域
    auto regions1 = createSpatialRegions(grid_dims1);
    auto regions2 = createSpatialRegions(grid_dims2);
    
    std::vector<RegionMatchPair> reliable_pairs;
    
    // 并行处理区域特征提取和匹配
    std::vector<LocalRegionFeature> features1(regions1.size());
    std::vector<LocalRegionFeature> features2(regions2.size());
    
    #pragma omp parallel for schedule(dynamic) if(regions1.size() > 4)
    for (size_t i = 0; i < regions1.size(); ++i) {
        features1[i] = extractRegionFeature(coarse_status1, regions1[i]);
    }
    
    #pragma omp parallel for schedule(dynamic) if(regions2.size() > 4)
    for (size_t i = 0; i < regions2.size(); ++i) {
        features2[i] = extractRegionFeature(coarse_status2, regions2[i]);
    }
    
    // 寻找可靠的区域匹配对
    for (size_t i = 0; i < features1.size(); ++i) {
        if (features1[i].quality_score < 0.3f) continue;  // 跳过低质量区域
        
        float best_similarity = 0.0f;
        int best_match = -1;
        
        for (size_t j = 0; j < features2.size(); ++j) {
            if (features2[j].quality_score < 0.3f) continue;
            
            float similarity = calculateFeatureSimilarity(features1[i], features2[j]);
            if (similarity > best_similarity && similarity > params_.similarity_threshold) {
                best_similarity = similarity;
                best_match = static_cast<int>(j);
            }
        }
        
        if (best_match >= 0) {
            // 检测这对区域的时间偏移
            float confidence;
            int offset = detectOffsetBetweenFeatures(features1[i], features2[best_match], confidence);
            
            reliable_pairs.emplace_back(static_cast<int>(i), best_match, 
                                      best_similarity, confidence, offset);
        }
    }
    
    // 检查是否有足够的可靠区域对
    if (reliable_pairs.size() < static_cast<size_t>(params_.min_reliable_regions)) {
        std::cout << "时间对齐：可靠区域对不足 (" << reliable_pairs.size() 
                  << " < " << params_.min_reliable_regions << ")" << std::endl;
        return result;
    }
    
    // 使用RANSAC进行鲁棒偏移估计
    int detected_offset = robustOffsetEstimation(reliable_pairs);
    
    // 精检测：如果使用了粗检测，在原分辨率上精确检测
    if (params_.use_coarse_detection && motion_status1.cols > 100) {
        detected_offset *= params_.coarse_downsample_factor;
        
        // 在detected_offset附近的小范围内精确搜索
        int fine_search_range = params_.coarse_downsample_factor;
        int best_fine_offset = detected_offset;
        float best_confidence = 0.0f;
        
        for (int fine_offset = detected_offset - fine_search_range; 
             fine_offset <= detected_offset + fine_search_range; ++fine_offset) {
            
            if (std::abs(fine_offset) > params_.max_time_offset) continue;
            
            // 在原分辨率下验证这个偏移
            cv::Mat adjusted_status2 = applyTimeOffset(motion_status2, fine_offset);
            
            // 计算全局相似性作为置信度
            float total_similarity = 0.0f;
            int valid_regions = 0;
            
            auto full_regions1 = createSpatialRegions(GridDimensions(motion_status1));
            auto full_regions2 = createSpatialRegions(GridDimensions(adjusted_status2));
            
            for (size_t i = 0; i < std::min(full_regions1.size(), full_regions2.size()); ++i) {
                auto feat1 = extractRegionFeature(motion_status1, full_regions1[i]);
                auto feat2 = extractRegionFeature(adjusted_status2, full_regions2[i]);
                
                if (feat1.quality_score > 0.3f && feat2.quality_score > 0.3f) {
                    total_similarity += calculateFeatureSimilarity(feat1, feat2);
                    valid_regions++;
                }
            }
            
            if (valid_regions > 0) {
                float avg_similarity = total_similarity / valid_regions;
                if (avg_similarity > best_confidence) {
                    best_confidence = avg_similarity;
                    best_fine_offset = fine_offset;
                }
            }
        }
        
        detected_offset = best_fine_offset;
        result.confidence = best_confidence;
    } else {
        // 计算整体置信度
        float total_confidence = 0.0f;
        for (const auto& pair : reliable_pairs) {
            total_confidence += pair.confidence * pair.similarity_score;
        }
        result.confidence = total_confidence / reliable_pairs.size();
    }
    
    // 填充结果
    result.detected_offset = detected_offset;
    result.num_reliable_regions = static_cast<int>(reliable_pairs.size());
    result.region_pairs = std::move(reliable_pairs);
    result.is_valid = true;
    
    std::cout << "时间对齐检测完成：偏移=" << detected_offset 
              << "帧，置信度=" << result.confidence 
              << "，可靠区域=" << result.num_reliable_regions << std::endl;
    
    return result;
}

cv::Mat TimeAlignmentEngine::applyTimeOffset(const cv::Mat& motion_status, int offset) {
    if (offset == 0) {
        return motion_status;
    }
    
    int rows = motion_status.rows;
    int cols = motion_status.cols;
    
    cv::Mat adjusted_status = cv::Mat::zeros(rows, cols, CV_8U);
    
    if (offset > 0) {
        // 向右偏移（延迟）
        int valid_cols = std::max(0, cols - offset);
        if (valid_cols > 0) {
            cv::Rect src_rect(0, 0, valid_cols, rows);
            cv::Rect dst_rect(offset, 0, valid_cols, rows);
            motion_status(src_rect).copyTo(adjusted_status(dst_rect));
        }
    } else {
        // 向左偏移（提前）
        int abs_offset = std::abs(offset);
        int valid_cols = std::max(0, cols - abs_offset);
        if (valid_cols > 0) {
            cv::Rect src_rect(abs_offset, 0, valid_cols, rows);
            cv::Rect dst_rect(0, 0, valid_cols, rows);
            motion_status(src_rect).copyTo(adjusted_status(dst_rect));
        }
    }
    
    return adjusted_status;
}

std::vector<std::vector<int>> TimeAlignmentEngine::createSpatialRegions(const GridDimensions& grid_dims) {
    std::vector<std::vector<int>> regions;
    
    int region_size = std::max(1, grid_dims.cols / params_.region_grid_size);
    int stride = std::max(1, static_cast<int>(region_size * (1.0f - params_.region_overlap_ratio)));
    
    for (int r = 0; r <= grid_dims.rows - region_size; r += stride) {
        for (int c = 0; c <= grid_dims.cols - region_size; c += stride) {
            std::vector<int> region_grids;
            
            for (int dr = 0; dr < region_size; ++dr) {
                for (int dc = 0; dc < region_size; ++dc) {
                    int grid_row = r + dr;
                    int grid_col = c + dc;
                    
                    if (grid_row < grid_dims.rows && grid_col < grid_dims.cols) {
                        int grid_index = grid_row * grid_dims.cols + grid_col;
                        if (grid_index < grid_dims.total_grids) {
                            region_grids.push_back(grid_index);
                        }
                    }
                }
            }
            
            if (!region_grids.empty()) {
                regions.push_back(std::move(region_grids));
            }
        }
    }
    
    return regions;
}

LocalRegionFeature TimeAlignmentEngine::extractRegionFeature(const cv::Mat& motion_status, 
                                                            const std::vector<int>& region_grids) {
    LocalRegionFeature feature(motion_status.cols);
    
    // 计算激活率序列
    feature.activation_rate = calculateActivationRate(motion_status, region_grids);
    
    // 计算变化率序列
    feature.change_rate = calculateChangeRate(feature.activation_rate);
    
    // 计算持续性序列
    feature.persistence = calculatePersistence(feature.activation_rate, params_.smoothing_window);
    
    // 计算特征质量分数
    float variance = 0.0f;
    float mean = std::accumulate(feature.activation_rate.begin(), feature.activation_rate.end(), 0.0f) 
                 / feature.activation_rate.size();
    
    for (float val : feature.activation_rate) {
        variance += (val - mean) * (val - mean);
    }
    variance /= feature.activation_rate.size();
    
    // 质量分数基于方差（运动模式的丰富程度）
    feature.quality_score = std::min(1.0f, std::sqrt(variance) * 2.0f);
    
    return feature;
}

float TimeAlignmentEngine::calculateFeatureSimilarity(const LocalRegionFeature& feature1, 
                                                     const LocalRegionFeature& feature2) {
    if (feature1.activation_rate.size() != feature2.activation_rate.size()) {
        return 0.0f;
    }
    
    // 使用多特征加权相似性
    const float w1 = 0.5f, w2 = 0.3f, w3 = 0.2f;  // 权重
    
    // 激活率相似性（皮尔逊相关系数）
    float corr1 = static_cast<float>(calculateNormalizedCrossCorrelation(
        feature1.activation_rate, feature2.activation_rate, 0, feature1.activation_rate.size()));
    
    // 变化率相似性
    float corr2 = static_cast<float>(calculateNormalizedCrossCorrelation(
        feature1.change_rate, feature2.change_rate, 0, feature1.change_rate.size()));
    
    // 持续性相似性
    float corr3 = static_cast<float>(calculateNormalizedCrossCorrelation(
        feature1.persistence, feature2.persistence, 0, feature1.persistence.size()));
    
    return w1 * std::max(0.0f, corr1) + w2 * std::max(0.0f, corr2) + w3 * std::max(0.0f, corr3);
}

int TimeAlignmentEngine::detectOffsetBetweenFeatures(const LocalRegionFeature& feature1, 
                                                    const LocalRegionFeature& feature2, 
                                                    float& confidence) {
    confidence = 0.0f;
    int best_offset = 0;
    double best_correlation = -1.0;
    
    int seq_len = static_cast<int>(feature1.activation_rate.size());
    int search_range = std::min(params_.max_time_offset, seq_len / 4);
    
    for (int offset = -search_range; offset <= search_range; ++offset) {
        int compare_length = seq_len - std::abs(offset);
        if (compare_length < seq_len / 2) continue;  // 至少要有一半的重叠
        
        double correlation = calculateNormalizedCrossCorrelation(
            feature1.activation_rate, feature2.activation_rate, offset, compare_length);
        
        if (correlation > best_correlation) {
            best_correlation = correlation;
            best_offset = offset;
        }
    }
    
    confidence = static_cast<float>(std::max(0.0, best_correlation));
    return best_offset;
}

double TimeAlignmentEngine::calculateNormalizedCrossCorrelation(const std::vector<float>& seq1, 
                                                               const std::vector<float>& seq2, 
                                                               int offset, int compare_length) {
    if (seq1.size() != seq2.size() || compare_length <= 0) {
        return 0.0;
    }
    
    int start1 = std::max(0, -offset);
    int start2 = std::max(0, offset);
    
    // 计算均值
    double mean1 = 0.0, mean2 = 0.0;
    for (int i = 0; i < compare_length; ++i) {
        mean1 += seq1[start1 + i];
        mean2 += seq2[start2 + i];
    }
    mean1 /= compare_length;
    mean2 /= compare_length;
    
    // 计算归一化互相关
    double numerator = 0.0, denom1 = 0.0, denom2 = 0.0;
    for (int i = 0; i < compare_length; ++i) {
        double diff1 = seq1[start1 + i] - mean1;
        double diff2 = seq2[start2 + i] - mean2;
        numerator += diff1 * diff2;
        denom1 += diff1 * diff1;
        denom2 += diff2 * diff2;
    }
    
    if (denom1 == 0.0 || denom2 == 0.0) {
        return 0.0;
    }
    
    return numerator / (std::sqrt(denom1) * std::sqrt(denom2));
}

int TimeAlignmentEngine::robustOffsetEstimation(const std::vector<RegionMatchPair>& region_pairs) {
    if (region_pairs.empty()) {
        return 0;
    }
    
    // 简化版RANSAC：使用加权中位数
    std::vector<std::pair<int, float>> weighted_offsets;
    for (const auto& pair : region_pairs) {
        float weight = pair.similarity_score * pair.confidence;
        weighted_offsets.emplace_back(pair.detected_offset, weight);
    }
    
    // 按偏移值排序
    std::sort(weighted_offsets.begin(), weighted_offsets.end());
    
    // 计算加权中位数
    float total_weight = 0.0f;
    for (const auto& wo : weighted_offsets) {
        total_weight += wo.second;
    }
    
    float target_weight = total_weight / 2.0f;
    float cumulative_weight = 0.0f;
    
    for (const auto& wo : weighted_offsets) {
        cumulative_weight += wo.second;
        if (cumulative_weight >= target_weight) {
            return wo.first;
        }
    }
    
    return weighted_offsets.empty() ? 0 : weighted_offsets[weighted_offsets.size() / 2].first;
}

cv::Mat TimeAlignmentEngine::temporalDownsample(const cv::Mat& motion_status, int factor) {
    if (factor <= 1) {
        return motion_status;
    }
    
    int new_cols = motion_status.cols / factor;
    if (new_cols == 0) return motion_status;
    
    cv::Mat downsampled = cv::Mat::zeros(motion_status.rows, new_cols, CV_8U);
    
    for (int r = 0; r < motion_status.rows; ++r) {
        for (int c = 0; c < new_cols; ++c) {
            // 取factor个时间点的最大值
            int max_val = 0;
            for (int t = 0; t < factor && c * factor + t < motion_status.cols; ++t) {
                max_val = std::max(max_val, static_cast<int>(motion_status.at<uchar>(r, c * factor + t)));
            }
            downsampled.at<uchar>(r, c) = static_cast<uchar>(max_val);
        }
    }
    
    return downsampled;
}

std::vector<float> TimeAlignmentEngine::movingAverage(const std::vector<float>& sequence, int window_size) {
    if (sequence.empty() || window_size <= 0) {
        return sequence;
    }
    
    std::vector<float> smoothed(sequence.size());
    int half_window = window_size / 2;
    
    for (size_t i = 0; i < sequence.size(); ++i) {
        int start = std::max(0, static_cast<int>(i) - half_window);
        int end = std::min(static_cast<int>(sequence.size()), static_cast<int>(i) + half_window + 1);
        
        float sum = 0.0f;
        for (int j = start; j < end; ++j) {
            sum += sequence[j];
        }
        smoothed[i] = sum / (end - start);
    }
    
    return smoothed;
}

std::vector<float> TimeAlignmentEngine::calculateActivationRate(const cv::Mat& motion_status, 
                                                               const std::vector<int>& region_grids) {
    std::vector<float> activation_rate(motion_status.cols);
    
    for (int t = 0; t < motion_status.cols; ++t) {
        int active_count = 0;
        for (int grid_idx : region_grids) {
            if (grid_idx < motion_status.rows && motion_status.at<uchar>(grid_idx, t) > 0) {
                active_count++;
            }
        }
        activation_rate[t] = static_cast<float>(active_count) / region_grids.size();
    }
    
    return activation_rate;
}

std::vector<float> TimeAlignmentEngine::calculateChangeRate(const std::vector<float>& activation_rate) {
    std::vector<float> change_rate(activation_rate.size(), 0.0f);
    
    for (size_t i = 1; i < activation_rate.size(); ++i) {
        change_rate[i] = std::abs(activation_rate[i] - activation_rate[i-1]);
    }
    
    return change_rate;
}

std::vector<float> TimeAlignmentEngine::calculatePersistence(const std::vector<float>& activation_rate, 
                                                            int window_size) {
    return movingAverage(activation_rate, window_size);
}

} // namespace VideoMatcher 