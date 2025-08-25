#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <future>
#include "video_matcher.h"
#include "feature_extractor.h"
#include "match_quality_assessor.h"

namespace VideoMatcher {

// 双向匹配结果
struct BidirectionalMatchResult {
    std::vector<MatchTriplet> forward_matches;    // 正向匹配 (video1 -> video2)
    std::vector<MatchTriplet> backward_matches;   // 反向匹配 (video2 -> video1)
    std::vector<MatchTriplet> consistent_matches; // 一致匹配
    std::vector<MatchTriplet> conflicted_matches; // 冲突匹配
    
    // 一致性统计
    float consistency_ratio = 0.0f;              // 一致性比例
    float forward_coverage = 0.0f;               // 正向覆盖率
    float backward_coverage = 0.0f;              // 反向覆盖率
    float mutual_coverage = 0.0f;                // 互相覆盖率
    
    // 匹配强度分布
    std::vector<float> consistency_strengths;    // 一致匹配的强度分布
    std::vector<float> conflict_severities;      // 冲突的严重程度
    
    // 质量评估
    std::vector<QualityMetrics> forward_qualities;  // 正向匹配质量
    std::vector<QualityMetrics> backward_qualities; // 反向匹配质量
    std::vector<QualityMetrics> consistent_qualities; // 一致匹配质量
    
    // 计算统计信息
    void computeStatistics();
    
    // 生成匹配报告
    std::string generateReport() const;
    
    // 获取高质量一致匹配
    std::vector<MatchTriplet> getHighQualityMatches(float quality_threshold = 0.7f) const;
};

// 匹配冲突信息
struct MatchConflict {
    enum class ConflictType {
        ONE_TO_MANY,        // 一对多冲突
        MANY_TO_ONE,        // 多对一冲突
        CYCLE,              // 循环冲突
        INCONSISTENT_DISTANCE, // 距离不一致
        GEOMETRIC_VIOLATION    // 几何约束违反
    };
    
    ConflictType type;
    std::vector<MatchTriplet> involved_matches; // 涉及的匹配
    float severity_score;                       // 冲突严重程度
    std::string description;                    // 冲突描述
    std::vector<std::string> resolution_suggestions; // 解决建议
    
    MatchConflict(ConflictType t, const std::vector<MatchTriplet>& matches, float severity)
        : type(t), involved_matches(matches), severity_score(severity) {}
};

// 一致性验证器
class ConsistencyValidator {
public:
    struct ValidationResult {
        bool is_consistent = false;
        float consistency_score = 0.0f;
        std::vector<MatchConflict> detected_conflicts;
        std::string validation_summary;
    };
    
    // 验证匹配一致性
    ValidationResult validateConsistency(const std::vector<MatchTriplet>& forward_matches,
                                        const std::vector<MatchTriplet>& backward_matches);
    
    // 检测几何约束违反
    std::vector<MatchConflict> detectGeometricViolations(
        const std::vector<MatchTriplet>& matches,
        const cv::Size& grid_layout1,
        const cv::Size& grid_layout2);
    
    // 检测距离不一致
    std::vector<MatchConflict> detectDistanceInconsistencies(
        const std::vector<MatchTriplet>& forward_matches,
        const std::vector<MatchTriplet>& backward_matches,
        float tolerance = 0.1f);
    
    // 验证空间连续性
    bool validateSpatialContinuity(const std::vector<MatchTriplet>& matches,
                                  const cv::Size& grid_layout);
    
private:
    // 构建匹配图
    std::unordered_map<int, std::vector<int>> buildMatchGraph(
        const std::vector<MatchTriplet>& matches, bool forward = true);
    
    // 检测循环
    std::vector<std::vector<int>> detectCycles(
        const std::unordered_map<int, std::vector<int>>& graph);
    
    // 计算匹配距离不一致性
    float computeDistanceInconsistency(const MatchTriplet& forward_match,
                                      const MatchTriplet& backward_match);
};

// 冲突解决器
class ConflictResolver {
public:
    enum class ResolutionStrategy {
        KEEP_STRONGEST,     // 保留最强匹配
        KEEP_MOST_CONSISTENT, // 保留最一致匹配
        WEIGHTED_VOTING,    // 加权投票
        GEOMETRIC_PRIORITY, // 几何优先
        QUALITY_BASED,      // 基于质量
        ENSEMBLE_DECISION   // 集成决策
    };
    
    struct ResolutionConfig {
        ResolutionStrategy strategy = ResolutionStrategy::ENSEMBLE_DECISION;
        float distance_weight = 0.3f;
        float quality_weight = 0.4f;
        float consistency_weight = 0.3f;
        bool preserve_spatial_structure = true;
        bool enforce_uniqueness = true;
    };
    
    ConflictResolver(); // 默认构造函数
    explicit ConflictResolver(const ResolutionConfig& config);
    
    // 解决匹配冲突
    std::vector<MatchTriplet> resolveConflicts(
        const std::vector<MatchTriplet>& forward_matches,
        const std::vector<MatchTriplet>& backward_matches,
        const std::vector<QualityMetrics>& forward_qualities,
        const std::vector<QualityMetrics>& backward_qualities);
    
    // 解决特定类型冲突
    std::vector<MatchTriplet> resolveOneToManyConflicts(
        const std::vector<MatchTriplet>& conflicted_matches,
        const std::vector<QualityMetrics>& qualities);
    
    std::vector<MatchTriplet> resolveManyToOneConflicts(
        const std::vector<MatchTriplet>& conflicted_matches,
        const std::vector<QualityMetrics>& qualities);
    
    // 应用几何约束
    std::vector<MatchTriplet> applyGeometricConstraints(
        const std::vector<MatchTriplet>& matches,
        const cv::Size& grid_layout1,
        const cv::Size& grid_layout2);
    
    // 强制唯一性约束
    std::vector<MatchTriplet> enforceUniqueness(
        const std::vector<MatchTriplet>& matches,
        const std::vector<QualityMetrics>& qualities);
    
private:
    ResolutionConfig config_;
    
    // 计算匹配分数
    float computeMatchScore(const MatchTriplet& match,
                           const QualityMetrics& quality,
                           const std::vector<MatchTriplet>& context_matches);
    
    // 检查几何一致性
    bool checkGeometricConsistency(const MatchTriplet& match1,
                                  const MatchTriplet& match2,
                                  const cv::Size& grid_layout1,
                                  const cv::Size& grid_layout2);
};

// 匹配传播增强器
class MatchPropagationEnhancer {
public:
    struct PropagationConfig {
        int max_propagation_steps = 3;
        float propagation_decay = 0.8f;
        float confidence_threshold = 0.5f;
        bool use_geometric_constraints = true;
        bool enable_adaptive_threshold = true;
    };
    
    MatchPropagationEnhancer(); // 默认构造函数
    explicit MatchPropagationEnhancer(const PropagationConfig& config);
    
    // 增强匹配传播
    std::vector<MatchTriplet> enhanceMatchPropagation(
        const std::vector<MatchTriplet>& seed_matches,
        const cv::Mat& motion_status1,
        const cv::Mat& motion_status2,
        const std::vector<FeatureDescriptor>& features1,
        const std::vector<FeatureDescriptor>& features2,
        const cv::Size& grid_layout1,
        const cv::Size& grid_layout2);
    
    // 基于可信度的传播
    std::vector<MatchTriplet> propagateByConfidence(
        const std::vector<MatchTriplet>& matches,
        const std::vector<QualityMetrics>& qualities,
        const cv::Size& grid_layout1,
        const cv::Size& grid_layout2);
    
    // 基于特征相似性的传播
    std::vector<MatchTriplet> propagateBySimilarity(
        const std::vector<MatchTriplet>& matches,
        const std::vector<FeatureDescriptor>& features1,
        const std::vector<FeatureDescriptor>& features2,
        const cv::Size& grid_layout1,
        const cv::Size& grid_layout2);
    
private:
    PropagationConfig config_;
    
    // 计算传播候选
    std::vector<std::pair<int, int>> computePropagationCandidates(
        int source_grid1, int source_grid2,
        const cv::Size& layout1, const cv::Size& layout2,
        int step_radius);
    
    // 评估传播可信度
    float evaluatePropagationConfidence(
        const MatchTriplet& source_match,
        int candidate_grid1, int candidate_grid2,
        const std::vector<FeatureDescriptor>& features1,
        const std::vector<FeatureDescriptor>& features2);
};

// 双向匹配器主类
class BidirectionalMatcher {
public:
    struct MatchingConfig {
        // 基本匹配参数
        bool enable_parallel_matching = true;
        bool enable_quality_assessment = true;
        bool enable_conflict_resolution = true;
        bool enable_match_propagation = true;
        
        // 一致性参数
        float consistency_threshold = 0.7f;
        float quality_threshold = 0.6f;
        float distance_tolerance = 0.15f;
        
        // 几何约束
        bool enforce_geometric_constraints = true;
        bool preserve_topology = true;
        float max_geometric_distortion = 0.2f;
        
        // 性能参数
        int max_threads = 4;
        bool use_gpu_acceleration = false;
        size_t max_memory_usage = 2ULL * 1024 * 1024 * 1024; // 2GB
    };
    
    BidirectionalMatcher(); // 默认构造函数
    explicit BidirectionalMatcher(const MatchingConfig& config);
    
    // 主要双向匹配接口
    BidirectionalMatchResult performBidirectionalMatching(
        const cv::Mat& motion_status1,
        const cv::Mat& motion_status2,
        const std::vector<FeatureDescriptor>& features1,
        const std::vector<FeatureDescriptor>& features2,
        const Parameters& params);
    
    // 分层双向匹配
    std::vector<BidirectionalMatchResult> performHierarchicalBidirectionalMatching(
        const std::vector<cv::Mat>& motion_status_pyramid1,
        const std::vector<cv::Mat>& motion_status_pyramid2,
        const std::vector<std::vector<FeatureDescriptor>>& feature_pyramid1,
        const std::vector<std::vector<FeatureDescriptor>>& feature_pyramid2,
        const Parameters& params);
    
    // 单向匹配 (内部使用)
    std::vector<MatchTriplet> performUnidirectionalMatching(
        const cv::Mat& motion_status_source,
        const cv::Mat& motion_status_target,
        const std::vector<FeatureDescriptor>& features_source,
        const std::vector<FeatureDescriptor>& features_target,
        const Parameters& params,
        bool is_forward = true);
    
    // 一致性分析
    BidirectionalMatchResult analyzeConsistency(
        const std::vector<MatchTriplet>& forward_matches,
        const std::vector<MatchTriplet>& backward_matches,
        const std::vector<QualityMetrics>& forward_qualities,
        const std::vector<QualityMetrics>& backward_qualities);
    
    // 配置管理
    void setConfig(const MatchingConfig& config) { config_ = config; }
    const MatchingConfig& getConfig() const { return config_; }
    
    // 性能统计
    struct MatchingStats {
        std::chrono::milliseconds total_time{0};
        std::chrono::milliseconds forward_time{0};
        std::chrono::milliseconds backward_time{0};
        std::chrono::milliseconds consistency_time{0};
        std::chrono::milliseconds resolution_time{0};
        
        size_t total_forward_matches = 0;
        size_t total_backward_matches = 0;
        size_t consistent_matches = 0;
        size_t resolved_conflicts = 0;
        
        float average_consistency_ratio = 0.0f;
        float average_match_quality = 0.0f;
    };
    
    MatchingStats getStats() const { return stats_; }
    void resetStats();
    
    // 自适应参数调优
    void adaptParameters(const std::vector<BidirectionalMatchResult>& historical_results);
    
private:
    MatchingConfig config_;
    
    // 核心组件
    std::unique_ptr<ConsistencyValidator> consistency_validator_;
    std::unique_ptr<ConflictResolver> conflict_resolver_;
    std::unique_ptr<MatchPropagationEnhancer> propagation_enhancer_;
    std::unique_ptr<MatchQualityAssessment> quality_assessor_;
    
    // 性能统计
    mutable MatchingStats stats_;
    
    // 匹配历史 (用于自适应调优)
    std::vector<BidirectionalMatchResult> matching_history_;
    
    // 内部方法
    void initializeComponents();
    
    // 并行匹配执行
    std::pair<std::vector<MatchTriplet>, std::vector<MatchTriplet>> executeParallelMatching(
        const cv::Mat& motion_status1,
        const cv::Mat& motion_status2,
        const std::vector<FeatureDescriptor>& features1,
        const std::vector<FeatureDescriptor>& features2,
        const Parameters& params);
    
    // 质量评估
    std::pair<std::vector<QualityMetrics>, std::vector<QualityMetrics>> assessMatchQualities(
        const std::vector<MatchTriplet>& forward_matches,
        const std::vector<MatchTriplet>& backward_matches,
        const std::vector<FeatureDescriptor>& features1,
        const std::vector<FeatureDescriptor>& features2,
        const cv::Mat& motion_status1,
        const cv::Mat& motion_status2);
    
    // 找到一致匹配
    std::vector<MatchTriplet> findConsistentMatches(
        const std::vector<MatchTriplet>& forward_matches,
        const std::vector<MatchTriplet>& backward_matches,
        float tolerance = 0.1f);
    
    // 找到冲突匹配
    std::vector<MatchTriplet> findConflictedMatches(
        const std::vector<MatchTriplet>& forward_matches,
        const std::vector<MatchTriplet>& backward_matches);
    
    // 计算覆盖率
    std::tuple<float, float, float> computeCoverageRatios(
        const std::vector<MatchTriplet>& forward_matches,
        const std::vector<MatchTriplet>& backward_matches,
        const std::vector<MatchTriplet>& consistent_matches,
        size_t total_grids1, size_t total_grids2);
    
    // 自适应阈值调整
    void adjustThresholds(const BidirectionalMatchResult& result);
    
    // 内存管理
    void manageMemoryUsage();
    
    // GPU资源管理
    void initializeGPUResources();
    void releaseGPUResources();
};

// 匹配结果优化器
class MatchResultOptimizer {
public:
    // 全局优化
    std::vector<MatchTriplet> optimizeGlobalConsistency(
        const std::vector<MatchTriplet>& matches,
        const std::vector<QualityMetrics>& qualities,
        const cv::Size& grid_layout1,
        const cv::Size& grid_layout2);
    
    // 局部优化
    std::vector<MatchTriplet> optimizeLocalConsistency(
        const std::vector<MatchTriplet>& matches,
        const std::vector<QualityMetrics>& qualities,
        int optimization_radius = 2);
    
    // 基于图的优化
    std::vector<MatchTriplet> optimizeByGraphCuts(
        const std::vector<MatchTriplet>& matches,
        const std::vector<QualityMetrics>& qualities,
        const cv::Size& grid_layout1,
        const cv::Size& grid_layout2);
    
private:
    // 构建优化图
    struct OptimizationGraph {
        std::vector<std::vector<float>> node_costs;   // 节点代价
        std::vector<std::vector<float>> edge_costs;   // 边代价
        std::vector<std::pair<int, int>> edges;       // 边连接
    };
    
    OptimizationGraph buildOptimizationGraph(
        const std::vector<MatchTriplet>& matches,
        const std::vector<QualityMetrics>& qualities,
        const cv::Size& grid_layout1,
        const cv::Size& grid_layout2);
    
    // 图割算法
    std::vector<bool> performGraphCut(const OptimizationGraph& graph);
};

} // namespace VideoMatcher