#include "bidirectional_matcher.h"
#include "video_matcher.h"
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <future>
#include <iostream>
#include <sstream>

// 为 std::pair<int, int> 提供自定义哈希函数
namespace std {
    template <>
    struct hash<pair<int, int>> {
        size_t operator()(const pair<int, int>& p) const {
            return hash<int>()(p.first) ^ (hash<int>()(p.second) << 1);
        }
    };
}
#ifdef _OPENMP
#include <omp.h>
#endif

namespace VideoMatcher {

// BidirectionalMatchResult 实现
void BidirectionalMatchResult::computeStatistics() {
    if (forward_matches.empty() && backward_matches.empty()) {
        consistency_ratio = 0.0f;
        forward_coverage = 0.0f;
        backward_coverage = 0.0f;
        mutual_coverage = 0.0f;
        return;
    }
    
    // 计算一致性比例
    if (!forward_matches.empty() || !backward_matches.empty()) {
        size_t total_matches = forward_matches.size() + backward_matches.size();
        consistency_ratio = total_matches > 0 ? 
            static_cast<float>(consistent_matches.size() * 2) / total_matches : 0.0f;
    }
    
    // 计算覆盖率 (简化实现)
    std::unordered_set<int> forward_grids1, forward_grids2;
    std::unordered_set<int> backward_grids1, backward_grids2;
    std::unordered_set<int> consistent_grids1, consistent_grids2;
    
    for (const auto& match : forward_matches) {
        forward_grids1.insert(match.grid1);
        forward_grids2.insert(match.grid2);
    }
    
    for (const auto& match : backward_matches) {
        backward_grids2.insert(match.grid1);  // 注意：反向匹配中grid1和grid2的角色
        backward_grids1.insert(match.grid2);
    }
    
    for (const auto& match : consistent_matches) {
        consistent_grids1.insert(match.grid1);
        consistent_grids2.insert(match.grid2);
    }
    
    // 假设总网格数
    size_t total_grids1 = std::max({forward_grids1.size(), backward_grids1.size(), consistent_grids1.size()});
    size_t total_grids2 = std::max({forward_grids2.size(), backward_grids2.size(), consistent_grids2.size()});
    
    if (total_grids1 > 0) {
        forward_coverage = static_cast<float>(forward_grids1.size()) / total_grids1;
        backward_coverage = static_cast<float>(backward_grids1.size()) / total_grids1;
        mutual_coverage = static_cast<float>(consistent_grids1.size()) / total_grids1;
    }
    
    // 计算一致匹配强度分布
    consistency_strengths.clear();
    for (const auto& match : consistent_matches) {
        consistency_strengths.push_back(1.0f - match.distance);  // 假设距离越小强度越高
    }
    
    // 计算冲突严重程度
    conflict_severities.clear();
    for (const auto& match : conflicted_matches) {
        conflict_severities.push_back(match.distance);  // 距离作为冲突严重程度
    }
}

std::string BidirectionalMatchResult::generateReport() const {
    std::ostringstream report;
    report << std::fixed << std::setprecision(3);
    
    report << "=== 双向匹配结果报告 ===\n";
    report << "正向匹配数: " << forward_matches.size() << "\n";
    report << "反向匹配数: " << backward_matches.size() << "\n";
    report << "一致匹配数: " << consistent_matches.size() << "\n";
    report << "冲突匹配数: " << conflicted_matches.size() << "\n\n";
    
    report << "一致性统计:\n";
    report << "  一致性比例: " << consistency_ratio << "\n";
    report << "  正向覆盖率: " << forward_coverage << "\n";
    report << "  反向覆盖率: " << backward_coverage << "\n";
    report << "  互相覆盖率: " << mutual_coverage << "\n\n";
    
    if (!consistency_strengths.empty()) {
        float avg_strength = std::accumulate(consistency_strengths.begin(), consistency_strengths.end(), 0.0f) / consistency_strengths.size();
        report << "平均一致匹配强度: " << avg_strength << "\n";
    }
    
    if (!conflict_severities.empty()) {
        float avg_severity = std::accumulate(conflict_severities.begin(), conflict_severities.end(), 0.0f) / conflict_severities.size();
        report << "平均冲突严重程度: " << avg_severity << "\n";
    }
    
    return report.str();
}

std::vector<MatchTriplet> BidirectionalMatchResult::getHighQualityMatches(float quality_threshold) const {
    std::vector<MatchTriplet> high_quality;
    
    for (size_t i = 0; i < consistent_matches.size() && i < consistent_qualities.size(); ++i) {
        if (consistent_qualities[i].overall_confidence >= quality_threshold) {
            high_quality.push_back(consistent_matches[i]);
        }
    }
    
    return high_quality;
}

// ConsistencyValidator 实现
ConsistencyValidator::ValidationResult ConsistencyValidator::validateConsistency(
    const std::vector<MatchTriplet>& forward_matches,
    const std::vector<MatchTriplet>& backward_matches) {
    
    ValidationResult result;
    
    // 构建匹配图
    auto forward_graph = buildMatchGraph(forward_matches, true);
    auto backward_graph = buildMatchGraph(backward_matches, false);
    
    // 检测冲突
    result.detected_conflicts = detectDistanceInconsistencies(forward_matches, backward_matches);
    
    // 检测循环
    auto forward_cycles = detectCycles(forward_graph);
    auto backward_cycles = detectCycles(backward_graph);
    
    for (const auto& cycle : forward_cycles) {
        std::vector<MatchTriplet> cycle_matches;
        float severity = static_cast<float>(cycle.size()) / 10.0f;  // 循环越长越严重
        result.detected_conflicts.emplace_back(MatchConflict::ConflictType::CYCLE, cycle_matches, severity);
    }
    
    // 计算一致性分数
    size_t total_potential_conflicts = forward_matches.size() + backward_matches.size();
    result.consistency_score = total_potential_conflicts > 0 ? 
        1.0f - static_cast<float>(result.detected_conflicts.size()) / total_potential_conflicts : 1.0f;
    
    result.is_consistent = result.consistency_score > 0.7f && result.detected_conflicts.size() < 5;
    
    // 生成验证摘要
    std::ostringstream summary;
    summary << "检测到 " << result.detected_conflicts.size() << " 个冲突, "
            << "一致性分数: " << result.consistency_score;
    result.validation_summary = summary.str();
    
    return result;
}

std::unordered_map<int, std::vector<int>> ConsistencyValidator::buildMatchGraph(
    const std::vector<MatchTriplet>& matches, bool forward) {
    
    std::unordered_map<int, std::vector<int>> graph;
    
    for (const auto& match : matches) {
        if (forward) {
            graph[match.grid1].push_back(match.grid2);
        } else {
            graph[match.grid2].push_back(match.grid1);  // 反向图
        }
    }
    
    return graph;
}

std::vector<std::vector<int>> ConsistencyValidator::detectCycles(
    const std::unordered_map<int, std::vector<int>>& graph) {
    
    std::vector<std::vector<int>> cycles;
    std::unordered_set<int> visited;
    std::unordered_set<int> in_stack;
    std::vector<int> current_path;
    
    // 简化的DFS循环检测
    std::function<void(int)> dfs = [&](int node) {
        if (in_stack.find(node) != in_stack.end()) {
            // 找到循环
            auto cycle_start = std::find(current_path.begin(), current_path.end(), node);
            if (cycle_start != current_path.end()) {
                std::vector<int> cycle(cycle_start, current_path.end());
                cycle.push_back(node);
                cycles.push_back(cycle);
            }
            return;
        }
        
        if (visited.find(node) != visited.end()) {
            return;
        }
        
        visited.insert(node);
        in_stack.insert(node);
        current_path.push_back(node);
        
        auto it = graph.find(node);
        if (it != graph.end()) {
            for (int neighbor : it->second) {
                dfs(neighbor);
            }
        }
        
        in_stack.erase(node);
        current_path.pop_back();
    };
    
    for (const auto& pair : graph) {
        if (visited.find(pair.first) == visited.end()) {
            dfs(pair.first);
        }
    }
    
    return cycles;
}

std::vector<MatchConflict> ConsistencyValidator::detectDistanceInconsistencies(
    const std::vector<MatchTriplet>& forward_matches,
    const std::vector<MatchTriplet>& backward_matches,
    float tolerance) {
    
    std::vector<MatchConflict> conflicts;
    
    // 构建查找映射
    std::unordered_map<std::pair<int, int>, float, std::hash<std::pair<int, int>>> forward_map;
    std::unordered_map<std::pair<int, int>, float, std::hash<std::pair<int, int>>> backward_map;
    
    for (const auto& match : forward_matches) {
        forward_map[{match.grid1, match.grid2}] = match.distance;
    }
    
    for (const auto& match : backward_matches) {
        backward_map[{match.grid2, match.grid1}] = match.distance;  // 注意反向
    }
    
    // 检查距离不一致
    for (const auto& forward_pair : forward_map) {
        auto backward_it = backward_map.find(forward_pair.first);
        if (backward_it != backward_map.end()) {
            float distance_diff = std::abs(forward_pair.second - backward_it->second);
            if (distance_diff > tolerance) {
                std::vector<MatchTriplet> involved_matches = {
                    {forward_pair.first.first, forward_pair.first.second, forward_pair.second},
                    {backward_it->first.second, backward_it->first.first, backward_it->second}
                };
                
                conflicts.emplace_back(MatchConflict::ConflictType::INCONSISTENT_DISTANCE, 
                                     involved_matches, distance_diff);
            }
        }
    }
    
    return conflicts;
}

// ConflictResolver 实现
ConflictResolver::ConflictResolver() : config_{} {
}

ConflictResolver::ConflictResolver(const ResolutionConfig& config) : config_(config) {
}

std::vector<MatchTriplet> ConflictResolver::resolveConflicts(
    const std::vector<MatchTriplet>& forward_matches,
    const std::vector<MatchTriplet>& backward_matches,
    const std::vector<QualityMetrics>& forward_qualities,
    const std::vector<QualityMetrics>& backward_qualities) {
    
    std::vector<MatchTriplet> resolved_matches;
    
    // 合并所有匹配
    std::vector<MatchTriplet> all_matches;
    std::vector<QualityMetrics> all_qualities;
    
    all_matches.insert(all_matches.end(), forward_matches.begin(), forward_matches.end());
    all_matches.insert(all_matches.end(), backward_matches.begin(), backward_matches.end());
    
    all_qualities.insert(all_qualities.end(), forward_qualities.begin(), forward_qualities.end());
    all_qualities.insert(all_qualities.end(), backward_qualities.begin(), backward_qualities.end());
    
    if (config_.enforce_uniqueness) {
        resolved_matches = enforceUniqueness(all_matches, all_qualities);
    } else {
        resolved_matches = all_matches;
    }
    
    return resolved_matches;
}

std::vector<MatchTriplet> ConflictResolver::resolveOneToManyConflicts(
    const std::vector<MatchTriplet>& conflicted_matches,
    const std::vector<QualityMetrics>& qualities) {
    
    std::vector<MatchTriplet> resolved;
    
    // 按grid1分组
    std::unordered_map<int, std::vector<std::pair<MatchTriplet, QualityMetrics>>> groups;
    
    for (size_t i = 0; i < conflicted_matches.size() && i < qualities.size(); ++i) {
        groups[conflicted_matches[i].grid1].emplace_back(conflicted_matches[i], qualities[i]);
    }
    
    // 对每组选择最佳匹配
    for (const auto& group : groups) {
        if (group.second.empty()) continue;
        
        auto best_match = std::max_element(group.second.begin(), group.second.end(),
            [this](const auto& a, const auto& b) {
                float score_a = computeMatchScore(a.first, a.second, {});
                float score_b = computeMatchScore(b.first, b.second, {});
                return score_a < score_b;
            });
        
        resolved.push_back(best_match->first);
    }
    
    return resolved;
}

std::vector<MatchTriplet> ConflictResolver::enforceUniqueness(
    const std::vector<MatchTriplet>& matches,
    const std::vector<QualityMetrics>& qualities) {
    
    std::vector<MatchTriplet> unique_matches;
    std::unordered_set<int> used_grid1, used_grid2;
    
    // 按质量分数排序
    std::vector<size_t> indices(matches.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        if (a < qualities.size() && b < qualities.size()) {
            return qualities[a].overall_confidence > qualities[b].overall_confidence;
        }
        return matches[a].distance < matches[b].distance;
    });
    
    // 贪心选择不冲突的匹配
    for (size_t idx : indices) {
        const auto& match = matches[idx];
        
        if (used_grid1.find(match.grid1) == used_grid1.end() &&
            used_grid2.find(match.grid2) == used_grid2.end()) {
            
            unique_matches.push_back(match);
            used_grid1.insert(match.grid1);
            used_grid2.insert(match.grid2);
        }
    }
    
    return unique_matches;
}

float ConflictResolver::computeMatchScore(const MatchTriplet& match,
                                        const QualityMetrics& quality,
                                        const std::vector<MatchTriplet>& context_matches) {
    (void)context_matches;  // 抑制未使用参数警告
    
    float distance_score = 1.0f - std::min(1.0f, match.distance);
    float quality_score = quality.overall_confidence;
    float consistency_score = quality.temporal_consistency;
    
    return config_.distance_weight * distance_score +
           config_.quality_weight * quality_score +
           config_.consistency_weight * consistency_score;
}

// MatchPropagationEnhancer 实现
MatchPropagationEnhancer::MatchPropagationEnhancer() : config_{} {
}

MatchPropagationEnhancer::MatchPropagationEnhancer(const PropagationConfig& config) : config_(config) {
}

// BidirectionalMatcher 主类实现
BidirectionalMatcher::BidirectionalMatcher() : config_{} {
    initializeComponents();
    resetStats();
}

BidirectionalMatcher::BidirectionalMatcher(const MatchingConfig& config) : config_(config) {
    initializeComponents();
    resetStats();
}

void BidirectionalMatcher::initializeComponents() {
    consistency_validator_ = std::make_unique<ConsistencyValidator>();
    
    ConflictResolver::ResolutionConfig resolution_config;
    resolution_config.strategy = ConflictResolver::ResolutionStrategy::ENSEMBLE_DECISION;
    resolution_config.preserve_spatial_structure = config_.preserve_topology;
    resolution_config.enforce_uniqueness = true;
    conflict_resolver_ = std::make_unique<ConflictResolver>(resolution_config);
    
    if (config_.enable_quality_assessment) {
        QualityAssessmentConfig quality_config;
        quality_config.enable_motion_analysis = true;
        quality_config.enable_temporal_analysis = true;
        quality_config.enable_spatial_analysis = true;
        quality_assessor_ = std::make_unique<MatchQualityAssessment>(quality_config);
    }
}

BidirectionalMatchResult BidirectionalMatcher::performBidirectionalMatching(
    const cv::Mat& motion_status1,
    const cv::Mat& motion_status2,
    const std::vector<FeatureDescriptor>& features1,
    const std::vector<FeatureDescriptor>& features2,
    const Parameters& params) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    BidirectionalMatchResult result;
    
    // 执行双向匹配
    if (config_.enable_parallel_matching) {
        auto [forward_matches, backward_matches] = executeParallelMatching(
            motion_status1, motion_status2, features1, features2, params);
        result.forward_matches = forward_matches;
        result.backward_matches = backward_matches;
    } else {
        // 串行执行
        result.forward_matches = performUnidirectionalMatching(
            motion_status1, motion_status2, features1, features2, params, true);
        result.backward_matches = performUnidirectionalMatching(
            motion_status2, motion_status1, features2, features1, params, false);
    }
    
    // 质量评估
    if (config_.enable_quality_assessment && quality_assessor_) {
        auto [forward_qualities, backward_qualities] = assessMatchQualities(
            result.forward_matches, result.backward_matches,
            features1, features2, motion_status1, motion_status2);
        result.forward_qualities = forward_qualities;
        result.backward_qualities = backward_qualities;
    }
    
    // 一致性分析
    result = analyzeConsistency(result.forward_matches, result.backward_matches,
                               result.forward_qualities, result.backward_qualities);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // 更新统计
    stats_.total_time += duration;
    stats_.total_forward_matches += result.forward_matches.size();
    stats_.total_backward_matches += result.backward_matches.size();
    stats_.consistent_matches += result.consistent_matches.size();
    
    return result;
}

std::pair<std::vector<MatchTriplet>, std::vector<MatchTriplet>> BidirectionalMatcher::executeParallelMatching(
    const cv::Mat& motion_status1,
    const cv::Mat& motion_status2,
    const std::vector<FeatureDescriptor>& features1,
    const std::vector<FeatureDescriptor>& features2,
    const Parameters& params) {
    
    std::vector<MatchTriplet> forward_matches, backward_matches;
    
    // 使用std::async进行并行执行
    auto forward_future = std::async(std::launch::async, [&]() {
        return performUnidirectionalMatching(motion_status1, motion_status2, 
                                           features1, features2, params, true);
    });
    
    auto backward_future = std::async(std::launch::async, [&]() {
        return performUnidirectionalMatching(motion_status2, motion_status1, 
                                           features2, features1, params, false);
    });
    
    forward_matches = forward_future.get();
    backward_matches = backward_future.get();
    
    return {forward_matches, backward_matches};
}

std::vector<MatchTriplet> BidirectionalMatcher::performUnidirectionalMatching(
    const cv::Mat& motion_status_source,
    const cv::Mat& motion_status_target,
    const std::vector<FeatureDescriptor>& features_source,
    const std::vector<FeatureDescriptor>& features_target,
    const Parameters& params,
    bool is_forward) {
    
    (void)features_source;  // 抑制未使用参数警告
    (void)features_target;
    (void)is_forward;
    
    // 使用现有的分段匹配器进行单向匹配
    std::map<int, std::set<int>> empty_dict;  // 空的先验匹配字典
    
    auto matches = SegmentMatcher::findMatchingGridWithSegment(
        motion_status_source, motion_status_target, params, empty_dict,
        motion_status_source.cols, motion_status_source.cols, false);
    
    return matches;
}

BidirectionalMatchResult BidirectionalMatcher::analyzeConsistency(
    const std::vector<MatchTriplet>& forward_matches,
    const std::vector<MatchTriplet>& backward_matches,
    const std::vector<QualityMetrics>& forward_qualities,
    const std::vector<QualityMetrics>& backward_qualities) {
    
    BidirectionalMatchResult result;
    result.forward_matches = forward_matches;
    result.backward_matches = backward_matches;
    result.forward_qualities = forward_qualities;
    result.backward_qualities = backward_qualities;
    
    // 找到一致匹配
    result.consistent_matches = findConsistentMatches(forward_matches, backward_matches);
    
    // 找到冲突匹配
    result.conflicted_matches = findConflictedMatches(forward_matches, backward_matches);
    
    // 如果启用冲突解决
    if (config_.enable_conflict_resolution && conflict_resolver_) {
        auto resolved = conflict_resolver_->resolveConflicts(
            forward_matches, backward_matches, forward_qualities, backward_qualities);
        
        // 更新一致匹配为解决后的匹配
        result.consistent_matches = resolved;
    }
    
    // 填充一致匹配的质量评估
    result.consistent_qualities.clear();
    for (const auto& consistent_match : result.consistent_matches) {
        // 查找对应的质量评估
        QualityMetrics quality;
        
        for (size_t i = 0; i < forward_matches.size() && i < forward_qualities.size(); ++i) {
            if (forward_matches[i].grid1 == consistent_match.grid1 && 
                forward_matches[i].grid2 == consistent_match.grid2) {
                quality = forward_qualities[i];
                break;
            }
        }
        
        result.consistent_qualities.push_back(quality);
    }
    
    // 计算统计信息
    result.computeStatistics();
    
    return result;
}

std::vector<MatchTriplet> BidirectionalMatcher::findConsistentMatches(
    const std::vector<MatchTriplet>& forward_matches,
    const std::vector<MatchTriplet>& backward_matches,
    float tolerance) {
    
    std::vector<MatchTriplet> consistent;
    
    for (const auto& forward_match : forward_matches) {
        for (const auto& backward_match : backward_matches) {
            // 检查是否为一致匹配：forward(A->B) 且 backward(B->A)
            if (forward_match.grid1 == backward_match.grid2 && 
                forward_match.grid2 == backward_match.grid1) {
                
                float distance_diff = std::abs(forward_match.distance - backward_match.distance);
                if (distance_diff <= tolerance) {
                    // 使用较好的距离值
                    MatchTriplet consistent_match = forward_match;
                    consistent_match.distance = std::min(forward_match.distance, backward_match.distance);
                    consistent.push_back(consistent_match);
                }
            }
        }
    }
    
    return consistent;
}

std::vector<MatchTriplet> BidirectionalMatcher::findConflictedMatches(
    const std::vector<MatchTriplet>& forward_matches,
    const std::vector<MatchTriplet>& backward_matches) {
    
    std::vector<MatchTriplet> conflicted;
    
    // 查找在正向匹配中存在但在反向匹配中不一致的匹配
    for (const auto& forward_match : forward_matches) {
        bool found_consistent = false;
        
        for (const auto& backward_match : backward_matches) {
            if (forward_match.grid1 == backward_match.grid2 && 
                forward_match.grid2 == backward_match.grid1) {
                found_consistent = true;
                break;
            }
        }
        
        if (!found_consistent) {
            conflicted.push_back(forward_match);
        }
    }
    
    return conflicted;
}

std::pair<std::vector<QualityMetrics>, std::vector<QualityMetrics>> BidirectionalMatcher::assessMatchQualities(
    const std::vector<MatchTriplet>& forward_matches,
    const std::vector<MatchTriplet>& backward_matches,
    const std::vector<FeatureDescriptor>& features1,
    const std::vector<FeatureDescriptor>& features2,
    const cv::Mat& motion_status1,
    const cv::Mat& motion_status2) {
    
    std::vector<QualityMetrics> forward_qualities, backward_qualities;
    
    if (!quality_assessor_) {
        return {forward_qualities, backward_qualities};
    }
    
    // 评估正向匹配质量
    for (const auto& match : forward_matches) {
        FeatureDescriptor feat1, feat2;
        
        // 查找对应的特征 (简化实现)
        if (static_cast<size_t>(match.grid1) < features1.size()) {
            feat1 = features1[match.grid1];
        }
        if (static_cast<size_t>(match.grid2) < features2.size()) {
            feat2 = features2[match.grid2];
        }
        
        auto quality = quality_assessor_->assessMatch(match, feat1, feat2, 
                                                     motion_status1, motion_status2);
        forward_qualities.push_back(quality);
    }
    
    // 评估反向匹配质量
    for (const auto& match : backward_matches) {
        FeatureDescriptor feat1, feat2;
        
        if (static_cast<size_t>(match.grid1) < features2.size()) {  // 注意反向
            feat1 = features2[match.grid1];
        }
        if (static_cast<size_t>(match.grid2) < features1.size()) {
            feat2 = features1[match.grid2];
        }
        
        auto quality = quality_assessor_->assessMatch(match, feat1, feat2, 
                                                     motion_status2, motion_status1);
        backward_qualities.push_back(quality);
    }
    
    return {forward_qualities, backward_qualities};
}

void BidirectionalMatcher::resetStats() {
    stats_ = MatchingStats{};
}

} // namespace VideoMatcher
