#include "temporal_consistency_enforcer.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace VideoMatcher {

// MatchTrajectory 实现
void MatchTrajectory::computeStatistics() {
    trajectory_length = static_cast<int>(temporal_matches.size());
    
    if (temporal_matches.empty()) {
        trajectory_stability = 0.0f;
        temporal_coherence = 0.0f;
        average_confidence = 0.0f;
        confidence_variance = 0.0f;
        return;
    }
    
    // 计算平均置信度
    if (!confidence_trajectory.empty()) {
        average_confidence = std::accumulate(confidence_trajectory.begin(), confidence_trajectory.end(), 0.0f) / confidence_trajectory.size();
        
        // 计算置信度方差
        float variance = 0.0f;
        for (float conf : confidence_trajectory) {
            variance += (conf - average_confidence) * (conf - average_confidence);
        }
        confidence_variance = variance / confidence_trajectory.size();
    }
    
    // 计算轨迹稳定性
    trajectory_stability = computeStability();
    
    // 计算时间连贯性
    temporal_coherence = computeCoherence();
    
    // 计算缺失段数
    missing_segments = 0;
    for (bool present : presence_mask) {
        if (!present) missing_segments++;
    }
    
    // 设置时间范围
    if (!temporal_matches.empty()) {
        start_time = 0;
        end_time = trajectory_length - 1;
    }
    
    // 计算整体质量
    overall_quality = 0.4f * trajectory_stability + 0.4f * temporal_coherence + 0.2f * average_confidence;
}

float MatchTrajectory::computeStability() const {
    if (distance_trajectory.size() < 2) return 0.0f;
    
    // 计算距离变化的稳定性
    float total_variation = 0.0f;
    for (size_t i = 1; i < distance_trajectory.size(); ++i) {
        total_variation += std::abs(distance_trajectory[i] - distance_trajectory[i-1]);
    }
    
    float avg_variation = total_variation / (distance_trajectory.size() - 1);
    return 1.0f / (1.0f + avg_variation);
}

float MatchTrajectory::computeCoherence() const {
    if (temporal_matches.size() < 3) return 0.0f;
    
    // 计算网格位置变化的连贯性
    float position_coherence = 0.0f;
    int valid_transitions = 0;
    
    for (size_t i = 2; i < temporal_matches.size(); ++i) {
        // 计算连续三个时间点的位置变化
        int dx1 = temporal_matches[i-1].grid1 - temporal_matches[i-2].grid1;
        int dy1 = temporal_matches[i-1].grid2 - temporal_matches[i-2].grid2;
        
        int dx2 = temporal_matches[i].grid1 - temporal_matches[i-1].grid1;
        int dy2 = temporal_matches[i].grid2 - temporal_matches[i-1].grid2;
        
        // 计算加速度 (二阶差分)
        int ddx = dx2 - dx1;
        int ddy = dy2 - dy1;
        float acceleration = std::sqrt(ddx * ddx + ddy * ddy);
        
        // 连贯性：加速度越小越连贯
        position_coherence += 1.0f / (1.0f + acceleration);
        valid_transitions++;
    }
    
    return valid_transitions > 0 ? position_coherence / valid_transitions : 0.0f;
}

bool MatchTrajectory::isReliable(float threshold) const {
    return overall_quality >= threshold && 
           trajectory_length >= 3 && 
           average_confidence >= threshold * 0.8f;
}

MatchTriplet MatchTrajectory::predictNextMatch(int time_step) const {
    if (temporal_matches.size() < 2) {
        return MatchTriplet{grid1_id, grid2_id, 1.0f};
    }
    
    // 简单的线性预测
    const auto& last = temporal_matches.back();
    const auto& second_last = temporal_matches[temporal_matches.size() - 2];
    
    int dx = last.grid1 - second_last.grid1;
    int dy = last.grid2 - second_last.grid2;
    
    float distance_trend = 0.0f;
    if (distance_trajectory.size() >= 2) {
        distance_trend = distance_trajectory.back() - distance_trajectory[distance_trajectory.size() - 2];
    }
    
    MatchTriplet predicted;
    predicted.grid1 = last.grid1 + dx;
    predicted.grid2 = last.grid2 + dy;
    predicted.distance = std::max(0.0f, last.distance + distance_trend);
    
    return predicted;
}

void MatchTrajectory::fillMissingSegments() {
    if (temporal_matches.size() < 2) return;
    
    // 简单的线性插值填充缺失段
    for (size_t i = 1; i < presence_mask.size() - 1; ++i) {
        if (!presence_mask[i]) {
            // 找到前后的有效点
            int prev = -1, next = -1;
            
            for (int j = static_cast<int>(i) - 1; j >= 0; --j) {
                if (presence_mask[j]) {
                    prev = j;
                    break;
                }
            }
            
            for (size_t j = i + 1; j < presence_mask.size(); ++j) {
                if (presence_mask[j]) {
                    next = static_cast<int>(j);
                    break;
                }
            }
            
            if (prev >= 0 && next >= 0 && 
                static_cast<size_t>(prev) < temporal_matches.size() && 
                static_cast<size_t>(next) < temporal_matches.size()) {
                
                // 线性插值
                float ratio = static_cast<float>(i - prev) / (next - prev);
                
                MatchTriplet interpolated;
                interpolated.grid1 = static_cast<int>(temporal_matches[prev].grid1 + 
                    ratio * (temporal_matches[next].grid1 - temporal_matches[prev].grid1));
                interpolated.grid2 = static_cast<int>(temporal_matches[prev].grid2 + 
                    ratio * (temporal_matches[next].grid2 - temporal_matches[prev].grid2));
                interpolated.distance = temporal_matches[prev].distance + 
                    ratio * (temporal_matches[next].distance - temporal_matches[prev].distance);
                
                // 插入到正确位置
                if (i < temporal_matches.size()) {
                    temporal_matches[i] = interpolated;
                } else {
                    temporal_matches.push_back(interpolated);
                }
                
                presence_mask[i] = true;
            }
        }
    }
}

void MatchTrajectory::smoothTrajectory(int window_size) {
    if (temporal_matches.size() < 3 || window_size < 3) return;
    
    std::vector<MatchTriplet> smoothed = temporal_matches;
    int half_window = window_size / 2;
    
    for (size_t i = half_window; i < temporal_matches.size() - half_window; ++i) {
        float avg_grid1 = 0, avg_grid2 = 0, avg_distance = 0;
        int count = 0;
        
        for (int j = -half_window; j <= half_window; ++j) {
            size_t idx = i + j;
            if (idx < temporal_matches.size()) {
                avg_grid1 += temporal_matches[idx].grid1;
                avg_grid2 += temporal_matches[idx].grid2;
                avg_distance += temporal_matches[idx].distance;
                count++;
            }
        }
        
        if (count > 0) {
            smoothed[i].grid1 = static_cast<int>(avg_grid1 / count);
            smoothed[i].grid2 = static_cast<int>(avg_grid2 / count);
            smoothed[i].distance = avg_distance / count;
        }
    }
    
    temporal_matches = smoothed;
}

// TrajectoryTracker 实现
TrajectoryTracker::TrajectoryTracker(const TemporalConstraintConfig& config) : config_(config) {
}

std::vector<MatchTrajectory> TrajectoryTracker::trackTrajectories(
    const std::vector<std::vector<MatchTriplet>>& hierarchical_matches,
    const std::vector<std::vector<QualityMetrics>>& hierarchical_qualities) {
    
    std::vector<MatchTrajectory> all_trajectories;
    
    if (hierarchical_matches.empty()) {
        return all_trajectories;
    }
    
    // 初始化轨迹
    if (!hierarchical_matches[0].empty()) {
        std::vector<QualityMetrics> initial_qualities;
        if (!hierarchical_qualities.empty() && !hierarchical_qualities[0].empty()) {
            initial_qualities = hierarchical_qualities[0];
        }
        
        active_trajectories_ = initializeTrajectories(hierarchical_matches[0], initial_qualities);
    }
    
    // 逐层关联轨迹
    for (size_t i = 1; i < hierarchical_matches.size(); ++i) {
        current_time_step_ = static_cast<int>(i);
        
        std::vector<QualityMetrics> current_qualities;
        if (i < hierarchical_qualities.size()) {
            current_qualities = hierarchical_qualities[i];
        }
        
        // 关联当前层的匹配到现有轨迹
        active_trajectories_ = associateTrajectories(active_trajectories_, 
                                                    hierarchical_matches[i], 
                                                    current_qualities);
        
        // 检测终止的轨迹
        auto terminated = detectTerminatedTrajectories(active_trajectories_, current_time_step_);
        completed_trajectories_.insert(completed_trajectories_.end(), terminated.begin(), terminated.end());
    }
    
    // 将所有轨迹（包括仍活跃的）添加到结果中
    all_trajectories.insert(all_trajectories.end(), completed_trajectories_.begin(), completed_trajectories_.end());
    all_trajectories.insert(all_trajectories.end(), active_trajectories_.begin(), active_trajectories_.end());
    
    // 计算每个轨迹的统计信息
    for (auto& trajectory : all_trajectories) {
        trajectory.computeStatistics();
    }
    
    return all_trajectories;
}

std::vector<MatchTrajectory> TrajectoryTracker::initializeTrajectories(
    const std::vector<MatchTriplet>& initial_matches,
    const std::vector<QualityMetrics>& initial_qualities) {
    
    std::vector<MatchTrajectory> trajectories;
    
    for (size_t i = 0; i < initial_matches.size(); ++i) {
        MatchTrajectory trajectory;
        trajectory.grid1_id = initial_matches[i].grid1;
        trajectory.grid2_id = initial_matches[i].grid2;
        trajectory.temporal_matches.push_back(initial_matches[i]);
        trajectory.distance_trajectory.push_back(initial_matches[i].distance);
        
        if (i < initial_qualities.size()) {
            trajectory.confidence_trajectory.push_back(initial_qualities[i].overall_confidence);
        } else {
            trajectory.confidence_trajectory.push_back(0.5f);
        }
        
        trajectory.presence_mask.push_back(true);
        
        trajectories.push_back(trajectory);
    }
    
    return trajectories;
}

std::vector<MatchTrajectory> TrajectoryTracker::associateTrajectories(
    const std::vector<MatchTrajectory>& previous_trajectories,
    const std::vector<MatchTriplet>& current_matches,
    const std::vector<QualityMetrics>& current_qualities) {
    
    std::vector<MatchTrajectory> updated_trajectories = previous_trajectories;
    
    // 执行轨迹关联
    auto associations = performTrajectoryAssociation(previous_trajectories, current_matches, current_qualities);
    
    // 更新已关联的轨迹
    std::vector<bool> match_used(current_matches.size(), false);
    
    for (const auto& association : associations) {
        int traj_idx = association.first;
        int match_idx = association.second;
        
        if (traj_idx >= 0 && traj_idx < static_cast<int>(updated_trajectories.size()) &&
            match_idx >= 0 && match_idx < static_cast<int>(current_matches.size())) {
            
            QualityMetrics quality;
            if (static_cast<size_t>(match_idx) < current_qualities.size()) {
                quality = current_qualities[match_idx];
            }
            
            updateTrajectory(updated_trajectories[traj_idx], current_matches[match_idx], quality, current_time_step_);
            match_used[match_idx] = true;
        }
    }
    
    // 为未关联的轨迹添加缺失标记
    for (auto& trajectory : updated_trajectories) {
        if (trajectory.presence_mask.size() <= static_cast<size_t>(current_time_step_)) {
            trajectory.presence_mask.resize(current_time_step_ + 1, false);
        }
    }
    
    // 为未使用的匹配创建新轨迹
    for (size_t i = 0; i < current_matches.size(); ++i) {
        if (!match_used[i]) {
            MatchTrajectory new_trajectory;
            new_trajectory.grid1_id = current_matches[i].grid1;
            new_trajectory.grid2_id = current_matches[i].grid2;
            new_trajectory.temporal_matches.push_back(current_matches[i]);
            new_trajectory.distance_trajectory.push_back(current_matches[i].distance);
            
            if (i < current_qualities.size()) {
                new_trajectory.confidence_trajectory.push_back(current_qualities[i].overall_confidence);
            } else {
                new_trajectory.confidence_trajectory.push_back(0.5f);
            }
            
            new_trajectory.presence_mask.resize(current_time_step_ + 1, false);
            new_trajectory.presence_mask[current_time_step_] = true;
            
            updated_trajectories.push_back(new_trajectory);
        }
    }
    
    return updated_trajectories;
}

std::vector<std::pair<int, int>> TrajectoryTracker::performTrajectoryAssociation(
    const std::vector<MatchTrajectory>& trajectories,
    const std::vector<MatchTriplet>& matches,
    const std::vector<QualityMetrics>& qualities) {
    
    std::vector<std::pair<int, int>> associations;
    
    // 计算关联代价矩阵
    std::vector<std::vector<float>> cost_matrix(trajectories.size(), 
                                               std::vector<float>(matches.size(), std::numeric_limits<float>::max()));
    
    for (size_t i = 0; i < trajectories.size(); ++i) {
        for (size_t j = 0; j < matches.size(); ++j) {
            QualityMetrics quality;
            if (j < qualities.size()) {
                quality = qualities[j];
            }
            
            cost_matrix[i][j] = computeAssociationCost(trajectories[i], matches[j], quality);
        }
    }
    
    // 简化的匈牙利算法或贪心匹配
    std::vector<bool> trajectory_used(trajectories.size(), false);
    std::vector<bool> match_used(matches.size(), false);
    
    // 贪心选择最低代价的关联
    for (int iteration = 0; iteration < std::min(trajectories.size(), matches.size()); ++iteration) {
        float min_cost = std::numeric_limits<float>::max();
        int best_traj = -1, best_match = -1;
        
        for (size_t i = 0; i < trajectories.size(); ++i) {
            if (trajectory_used[i]) continue;
            
            for (size_t j = 0; j < matches.size(); ++j) {
                if (match_used[j]) continue;
                
                if (cost_matrix[i][j] < min_cost) {
                    min_cost = cost_matrix[i][j];
                    best_traj = static_cast<int>(i);
                    best_match = static_cast<int>(j);
                }
            }
        }
        
        if (best_traj >= 0 && best_match >= 0 && min_cost < 1.0f) {  // 阈值过滤
            associations.emplace_back(best_traj, best_match);
            trajectory_used[best_traj] = true;
            match_used[best_match] = true;
        } else {
            break;  // 没有更好的关联
        }
    }
    
    return associations;
}

float TrajectoryTracker::computeAssociationCost(const MatchTrajectory& trajectory,
                                               const MatchTriplet& match,
                                               const QualityMetrics& quality) {
    if (trajectory.temporal_matches.empty()) {
        return std::numeric_limits<float>::max();
    }
    
    // 预测下一个位置
    MatchTriplet predicted = predictTrajectoryNext(trajectory);
    
    // 位置差异代价
    float position_cost = std::sqrt(std::pow(predicted.grid1 - match.grid1, 2) + 
                                   std::pow(predicted.grid2 - match.grid2, 2));
    
    // 距离差异代价
    float distance_cost = std::abs(predicted.distance - match.distance);
    
    // 质量代价
    float quality_cost = 1.0f - quality.overall_confidence;
    
    // 综合代价
    return 0.5f * position_cost + 0.3f * distance_cost + 0.2f * quality_cost;
}

MatchTriplet TrajectoryTracker::predictTrajectoryNext(const MatchTrajectory& trajectory) {
    return trajectory.predictNextMatch(current_time_step_ + 1);
}

void TrajectoryTracker::updateTrajectory(MatchTrajectory& trajectory,
                                        const MatchTriplet& new_match,
                                        const QualityMetrics& quality,
                                        int time_step) {
    trajectory.temporal_matches.push_back(new_match);
    trajectory.distance_trajectory.push_back(new_match.distance);
    trajectory.confidence_trajectory.push_back(quality.overall_confidence);
    
    // 确保presence_mask有足够的空间
    if (trajectory.presence_mask.size() <= static_cast<size_t>(time_step)) {
        trajectory.presence_mask.resize(time_step + 1, false);
    }
    trajectory.presence_mask[time_step] = true;
}

std::vector<MatchTrajectory> TrajectoryTracker::detectTerminatedTrajectories(
    const std::vector<MatchTrajectory>& active_trajectories,
    int current_time) {
    
    std::vector<MatchTrajectory> terminated;
    
    for (const auto& trajectory : active_trajectories) {
        // 检查轨迹是否在最近几个时间步中没有更新
        bool recently_updated = false;
        
        if (!trajectory.presence_mask.empty()) {
            int check_window = std::min(3, current_time);
            for (int i = std::max(0, current_time - check_window); i <= current_time; ++i) {
                if (static_cast<size_t>(i) < trajectory.presence_mask.size() && trajectory.presence_mask[i]) {
                    recently_updated = true;
                    break;
                }
            }
        }
        
        if (!recently_updated && trajectory.trajectory_length >= config_.min_trajectory_length) {
            terminated.push_back(trajectory);
        }
    }
    
    return terminated;
}

// TemporalSmoother 实现
TemporalSmoother::TemporalSmoother() : config_{} {
}

TemporalSmoother::TemporalSmoother(const SmoothingConfig& config) : config_(config) {
}

std::vector<MatchTriplet> TemporalSmoother::smoothMatches(
    const std::vector<MatchTriplet>& raw_matches,
    const std::vector<QualityMetrics>& qualities) {
    
    if (raw_matches.size() < 3) {
        return raw_matches;
    }
    
    std::vector<MatchTriplet> smoothed = raw_matches;
    
    // 提取各个分量进行平滑
    std::vector<float> grid1_values, grid2_values, distance_values;
    std::vector<float> weights;
    
    for (size_t i = 0; i < raw_matches.size(); ++i) {
        grid1_values.push_back(static_cast<float>(raw_matches[i].grid1));
        grid2_values.push_back(static_cast<float>(raw_matches[i].grid2));
        distance_values.push_back(raw_matches[i].distance);
        
        float weight = 1.0f;
        if (i < qualities.size()) {
            weight = qualities[i].overall_confidence;
        }
        weights.push_back(weight);
    }
    
    // 根据配置的方法进行平滑
    std::vector<float> smoothed_grid1, smoothed_grid2, smoothed_distances;
    
    switch (config_.method) {
        case SmoothingMethod::GAUSSIAN_FILTER:
            smoothed_grid1 = applyGaussianFilter(grid1_values, config_.sigma);
            smoothed_grid2 = applyGaussianFilter(grid2_values, config_.sigma);
            smoothed_distances = applyGaussianFilter(distance_values, config_.sigma);
            break;
            
        case SmoothingMethod::MEDIAN_FILTER:
            smoothed_grid1 = applyMedianFilter(grid1_values, config_.window_size);
            smoothed_grid2 = applyMedianFilter(grid2_values, config_.window_size);
            smoothed_distances = applyMedianFilter(distance_values, config_.window_size);
            break;
            
        case SmoothingMethod::ADAPTIVE_FILTER:
        default:
            smoothed_grid1 = applyKalmanFilter(grid1_values, weights);
            smoothed_grid2 = applyKalmanFilter(grid2_values, weights);
            smoothed_distances = applyKalmanFilter(distance_values, weights);
            break;
    }
    
    // 重构平滑后的匹配
    for (size_t i = 0; i < smoothed.size(); ++i) {
        if (i < smoothed_grid1.size()) {
            smoothed[i].grid1 = static_cast<int>(std::round(smoothed_grid1[i]));
        }
        if (i < smoothed_grid2.size()) {
            smoothed[i].grid2 = static_cast<int>(std::round(smoothed_grid2[i]));
        }
        if (i < smoothed_distances.size()) {
            smoothed[i].distance = smoothed_distances[i];
        }
    }
    
    return smoothed;
}

std::vector<float> TemporalSmoother::applyGaussianFilter(const std::vector<float>& signal, float sigma) {
    if (signal.empty()) return signal;
    
    std::vector<float> filtered = signal;
    int window_size = static_cast<int>(6 * sigma);  // 6sigma窗口
    if (window_size < 3) window_size = 3;
    if (window_size % 2 == 0) window_size++;  // 确保奇数
    
    int half_window = window_size / 2;
    
    // 生成高斯核
    std::vector<float> kernel(window_size);
    float sum = 0.0f;
    for (int i = 0; i < window_size; ++i) {
        float x = i - half_window;
        kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    
    // 归一化核
    for (float& k : kernel) {
        k /= sum;
    }
    
    // 应用滤波
    for (size_t i = 0; i < signal.size(); ++i) {
        float filtered_value = 0.0f;
        float weight_sum = 0.0f;
        
        for (int j = -half_window; j <= half_window; ++j) {
            int idx = static_cast<int>(i) + j;
            if (idx >= 0 && idx < static_cast<int>(signal.size())) {
                float weight = kernel[j + half_window];
                filtered_value += signal[idx] * weight;
                weight_sum += weight;
            }
        }
        
        if (weight_sum > 0) {
            filtered[i] = filtered_value / weight_sum;
        }
    }
    
    return filtered;
}

std::vector<float> TemporalSmoother::applyMedianFilter(const std::vector<float>& signal, int window_size) {
    if (signal.empty()) return signal;
    
    std::vector<float> filtered = signal;
    int half_window = window_size / 2;
    
    for (size_t i = 0; i < signal.size(); ++i) {
        std::vector<float> window_values;
        
        for (int j = -half_window; j <= half_window; ++j) {
            int idx = static_cast<int>(i) + j;
            if (idx >= 0 && idx < static_cast<int>(signal.size())) {
                window_values.push_back(signal[idx]);
            }
        }
        
        if (!window_values.empty()) {
            std::sort(window_values.begin(), window_values.end());
            filtered[i] = window_values[window_values.size() / 2];
        }
    }
    
    return filtered;
}

std::vector<float> TemporalSmoother::applyKalmanFilter(const std::vector<float>& signal, 
                                                      const std::vector<float>& confidence) {
    if (signal.empty()) return signal;
    
    std::vector<float> filtered = signal;
    
    // 简化的卡尔曼滤波器
    float estimated_state = signal[0];
    float estimation_error = 1.0f;
    float process_noise = 0.1f;
    
    for (size_t i = 1; i < signal.size(); ++i) {
        // 预测步骤
        float predicted_state = estimated_state;
        float predicted_error = estimation_error + process_noise;
        
        // 测量噪声基于置信度
        float measurement_noise = 1.0f;
        if (i < confidence.size()) {
            measurement_noise = 1.0f - confidence[i];
        }
        
        // 更新步骤
        float kalman_gain = predicted_error / (predicted_error + measurement_noise);
        estimated_state = predicted_state + kalman_gain * (signal[i] - predicted_state);
        estimation_error = (1.0f - kalman_gain) * predicted_error;
        
        filtered[i] = estimated_state;
    }
    
    return filtered;
}

// TemporalConsistencyEnforcer 主类实现
TemporalConsistencyEnforcer::TemporalConsistencyEnforcer(const TemporalConstraintConfig& config) 
    : config_(config) {
    initializeComponents();
    resetStats();
}

void TemporalConsistencyEnforcer::initializeComponents() {
    trajectory_tracker_ = std::make_unique<TrajectoryTracker>(config_);
    
    TemporalSmoother::SmoothingConfig smoothing_config;
    smoothing_config.window_size = config_.smoothing_window_size;
    smoothing_config.preserve_edges = true;
    temporal_smoother_ = std::make_unique<TemporalSmoother>(smoothing_config);
    
    // 其他组件的初始化...
}

std::vector<MatchTriplet> TemporalConsistencyEnforcer::enforceTemporalConsistency(
    const std::vector<MatchTriplet>& raw_matches,
    const std::vector<QualityMetrics>& qualities,
    const cv::Mat& motion_status1,
    const cv::Mat& motion_status2,
    int temporal_window) {
    
    (void)motion_status1;  // 抑制未使用参数警告
    (void)motion_status2;
    (void)temporal_window;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<MatchTriplet> consistent_matches = raw_matches;
    
    // 时间平滑
    if (config_.enable_temporal_smoothing && temporal_smoother_) {
        consistent_matches = temporal_smoother_->smoothMatches(consistent_matches, qualities);
        stats_.total_matches_processed += consistent_matches.size();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    stats_.processing_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    return consistent_matches;
}

std::vector<MatchTrajectory> TemporalConsistencyEnforcer::trackMatchTrajectories(
    const std::vector<std::vector<MatchTriplet>>& hierarchical_matches,
    const std::vector<std::vector<QualityMetrics>>& hierarchical_qualities) {
    
    if (trajectory_tracker_) {
        auto trajectories = trajectory_tracker_->trackTrajectories(hierarchical_matches, hierarchical_qualities);
        stats_.trajectories_tracked += trajectories.size();
        
        // 计算平均轨迹质量
        if (!trajectories.empty()) {
            float total_quality = 0.0f;
            for (const auto& traj : trajectories) {
                total_quality += traj.overall_quality;
            }
            stats_.average_trajectory_quality = total_quality / trajectories.size();
        }
        
        return trajectories;
    }
    
    return {};
}

void TemporalConsistencyEnforcer::resetStats() {
    stats_ = ConsistencyStats{};
}

float TemporalConsistencyEnforcer::assessTemporalQuality(
    const std::vector<MatchTriplet>& matches,
    const std::vector<QualityMetrics>& qualities) {
    
    if (matches.empty()) return 0.0f;
    
    // 评估时间一致性质量
    float consistency_score = 0.0f;
    
    // 距离变化的平滑性
    if (matches.size() > 1) {
        float distance_variation = 0.0f;
        for (size_t i = 1; i < matches.size(); ++i) {
            distance_variation += std::abs(matches[i].distance - matches[i-1].distance);
        }
        distance_variation /= (matches.size() - 1);
        consistency_score += 0.5f / (1.0f + distance_variation);
    }
    
    // 质量变化的平滑性
    if (qualities.size() > 1) {
        float quality_variation = 0.0f;
        for (size_t i = 1; i < qualities.size(); ++i) {
            quality_variation += std::abs(qualities[i].overall_confidence - qualities[i-1].overall_confidence);
        }
        quality_variation /= (qualities.size() - 1);
        consistency_score += 0.5f / (1.0f + quality_variation);
    }
    
    return consistency_score;
}

// MissingDataInterpolator 实现
MissingDataInterpolator::MissingDataInterpolator() : config_{} {
}

MissingDataInterpolator::MissingDataInterpolator(const InterpolationConfig& config) : config_(config) {
}

} // namespace VideoMatcher
