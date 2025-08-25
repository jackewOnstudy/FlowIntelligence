#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <chrono>
#include "video_matcher.h"
#include "feature_extractor.h"

namespace VideoMatcher {

// 匹配质量指标
struct QualityMetrics {
    // 核心质量指标
    float motion_coherence = 0.0f;      // 运动连贯性 [0,1]
    float temporal_consistency = 0.0f;  // 时间一致性 [0,1]
    float spatial_continuity = 0.0f;    // 空间连续性 [0,1]
    float feature_reliability = 0.0f;   // 特征可靠性 [0,1]
    float geometric_similarity = 0.0f;  // 几何相似性 [0,1]
    
    // 统计质量指标
    float cross_correlation = 0.0f;     // 互相关系数 [-1,1]
    float mutual_information = 0.0f;    // 互信息 [0,∞]
    float structural_similarity = 0.0f; // 结构相似性 [0,1]
    
    // 综合质量分数
    float overall_confidence = 0.0f;    // 综合置信度 [0,1]
    float match_strength = 0.0f;        // 匹配强度 [0,1]
    float uniqueness_score = 0.0f;      // 唯一性分数 [0,1]
    
    // 辅助信息
    int evaluation_window = 0;          // 评估窗口大小
    float noise_level = 0.0f;           // 噪声水平估计
    std::chrono::milliseconds compute_time; // 计算耗时
    
    // 构造函数
    QualityMetrics() = default;
    
    // 质量等级判断
    enum class QualityLevel {
        EXCELLENT,  // 优秀 (>0.8)
        GOOD,       // 良好 (0.6-0.8)
        FAIR,       // 一般 (0.4-0.6)
        POOR,       // 较差 (0.2-0.4)
        BAD         // 很差 (<0.2)
    };
    
    QualityLevel getQualityLevel() const;
    
    // 是否为可靠匹配
    bool isReliable(float threshold = 0.6f) const;
    
    // 转换为向量形式 (用于机器学习)
    std::vector<float> toVector() const;
    
    // 从向量恢复
    void fromVector(const std::vector<float>& vec);
    
    // 打印质量报告
    std::string generateReport() const;
};

// 质量评估配置
struct QualityAssessmentConfig {
    // 评估方法选择
    bool enable_motion_analysis = true;
    bool enable_temporal_analysis = true;
    bool enable_spatial_analysis = true;
    bool enable_feature_analysis = true;
    bool enable_statistical_analysis = true;
    
    // 评估参数
    int temporal_window_size = 16;      // 时间窗口大小
    int spatial_neighbor_radius = 2;    // 空间邻域半径
    float noise_threshold = 0.1f;       // 噪声阈值
    float consistency_threshold = 0.7f; // 一致性阈值
    
    // 权重配置
    struct WeightConfig {
        float motion_weight = 0.25f;
        float temporal_weight = 0.25f;
        float spatial_weight = 0.20f;
        float feature_weight = 0.20f;
        float statistical_weight = 0.10f;
    } weights;
    
    // 性能配置
    bool use_parallel_processing = true;
    bool enable_gpu_acceleration = false;
    int max_threads = 4;
    
    QualityAssessmentConfig() = default;
};

// 运动连贯性分析器
class MotionCoherenceAnalyzer {
public:
    struct CoherenceResult {
        float direction_consistency;    // 方向一致性
        float magnitude_stability;     // 幅度稳定性
        float velocity_smoothness;     // 速度平滑性
        float acceleration_bounds;     // 加速度边界
        float overall_coherence;       // 整体连贯性
    };
    
    explicit MotionCoherenceAnalyzer(int analysis_window = 8);
    
    // 分析运动连贯性
    CoherenceResult analyzeCoherence(const std::vector<cv::Mat>& motion_fields,
                                   const cv::Rect& region1,
                                   const cv::Rect& region2);
    
    // 分析运动序列连贯性
    CoherenceResult analyzeSequenceCoherence(const std::vector<bool>& sequence1,
                                            const std::vector<bool>& sequence2);
    
    // 评估运动模式相似性
    float assessMotionPatternSimilarity(const std::vector<float>& pattern1,
                                       const std::vector<float>& pattern2);
    
private:
    int analysis_window_;
    
    // 计算方向一致性
    float computeDirectionConsistency(const std::vector<cv::Point2f>& velocities);
    
    // 计算幅度稳定性
    float computeMagnitudeStability(const std::vector<float>& magnitudes);
    
    // 计算速度平滑性
    float computeVelocitySmoothness(const std::vector<cv::Point2f>& velocities);
};

// 时间一致性分析器
class TemporalConsistencyAnalyzer {
public:
    struct ConsistencyResult {
        float temporal_correlation;    // 时间相关性
        float phase_alignment;        // 相位对齐度
        float frequency_matching;     // 频率匹配度
        float trend_similarity;       // 趋势相似性
        float periodicity_match;      // 周期性匹配
        float overall_consistency;    // 整体一致性
    };
    
    explicit TemporalConsistencyAnalyzer(int window_size = 32);
    
    // 分析时间一致性
    ConsistencyResult analyzeConsistency(const std::vector<float>& signal1,
                                        const std::vector<float>& signal2);
    
    // 分析布尔序列一致性
    ConsistencyResult analyzeBooleanConsistency(const std::vector<bool>& sequence1,
                                               const std::vector<bool>& sequence2);
    
    // 计算时间偏移
    int estimateTimeOffset(const std::vector<float>& signal1,
                          const std::vector<float>& signal2,
                          int max_offset = 10);
    
    // 评估周期性
    float assessPeriodicity(const std::vector<float>& signal);
    
private:
    int window_size_;
    cv::Mat fft_buffer1_, fft_buffer2_;
    
    // 计算互相关
    std::vector<float> computeCrossCorrelation(const std::vector<float>& sig1,
                                              const std::vector<float>& sig2,
                                              int max_lag);
    
    // FFT相位分析
    float computePhaseAlignment(const std::vector<float>& sig1,
                               const std::vector<float>& sig2);
    
    // 频域匹配分析
    float computeFrequencyMatching(const std::vector<float>& sig1,
                                  const std::vector<float>& sig2);
};

// 空间连续性分析器
class SpatialContinuityAnalyzer {
public:
    struct ContinuityResult {
        float neighbor_consistency;   // 邻域一致性
        float gradient_smoothness;    // 梯度平滑性
        float boundary_coherence;     // 边界连贯性
        float texture_continuity;     // 纹理连续性
        float overall_continuity;     // 整体连续性
    };
    
    explicit SpatialContinuityAnalyzer(int neighbor_radius = 2);
    
    // 分析空间连续性
    ContinuityResult analyzeContinuity(const cv::Mat& motion_status1,
                                      const cv::Mat& motion_status2,
                                      const std::vector<MatchTriplet>& matches,
                                      const cv::Size& grid_layout);
    
    // 分析邻域一致性
    float analyzeNeighborConsistency(const std::vector<MatchTriplet>& matches,
                                    const cv::Size& grid_layout,
                                    int target_grid);
    
    // 计算空间自相关
    float computeSpatialAutocorrelation(const cv::Mat& motion_field,
                                       const cv::Rect& region);
    
private:
    int neighbor_radius_;
    
    // 获取网格邻居
    std::vector<int> getGridNeighbors(int grid_id, const cv::Size& layout, int radius);
    
    // 计算梯度平滑性
    float computeGradientSmoothness(const cv::Mat& field, const cv::Rect& region);
    
    // 分析边界连贯性
    float analyzeBoundaryCoherence(const cv::Mat& field, const cv::Rect& region);
};

// 特征可靠性评估器
class FeatureReliabilityAssessor {
public:
    struct ReliabilityResult {
        float feature_stability;      // 特征稳定性
        float discriminative_power;   // 判别能力
        float noise_robustness;       // 噪声鲁棒性
        float temporal_persistence;   // 时间持续性
        float overall_reliability;    // 整体可靠性
    };
    
    FeatureReliabilityAssessor();
    
    // 评估特征可靠性
    ReliabilityResult assessReliability(const FeatureDescriptor& feature1,
                                       const FeatureDescriptor& feature2,
                                       const std::vector<FeatureDescriptor>& context_features);
    
    // 评估单个特征质量
    float assessSingleFeatureQuality(const FeatureDescriptor& feature);
    
    // 计算特征判别能力
    float computeDiscriminativePower(const FeatureDescriptor& feature,
                                    const std::vector<FeatureDescriptor>& background_features);
    
    // 噪声鲁棒性测试
    float testNoiseRobustness(const FeatureDescriptor& feature,
                             float noise_level = 0.1f);
    
private:
    // 计算特征稳定性
    float computeFeatureStability(const FeatureDescriptor& feature);
    
    // 评估时间持续性
    float assessTemporalPersistence(const FeatureDescriptor& feature);
};

// 统计质量分析器
class StatisticalQualityAnalyzer {
public:
    struct StatisticalResult {
        float cross_correlation;      // 互相关系数
        float mutual_information;     // 互信息
        float structural_similarity;  // 结构相似性
        float kl_divergence;          // KL散度
        float wasserstein_distance;   // Wasserstein距离
        float overall_statistical_score; // 统计质量分数
    };
    
    StatisticalQualityAnalyzer();
    
    // 综合统计分析
    StatisticalResult performStatisticalAnalysis(const std::vector<float>& data1,
                                                 const std::vector<float>& data2);
    
    // 计算互信息
    float computeMutualInformation(const std::vector<float>& x,
                                  const std::vector<float>& y,
                                  int bins = 32);
    
    // 计算结构相似性
    float computeStructuralSimilarity(const cv::Mat& img1, const cv::Mat& img2);
    
    // 计算KL散度
    float computeKLDivergence(const std::vector<float>& p,
                             const std::vector<float>& q);
    
    // 计算Wasserstein距离
    float computeWassersteinDistance(const std::vector<float>& x,
                                    const std::vector<float>& y);
    
private:
    // 估计概率密度函数
    std::vector<float> estimatePDF(const std::vector<float>& data, int bins);
    
    // 计算熵
    float computeEntropy(const std::vector<float>& probabilities);
    
    // 计算联合熵
    float computeJointEntropy(const std::vector<std::vector<float>>& joint_probabilities);
};

// 主质量评估器
class MatchQualityAssessment {
public:
    // 构造函数
    explicit MatchQualityAssessment(const QualityAssessmentConfig& config = QualityAssessmentConfig{});
    
    // 主要评估接口
    QualityMetrics assessMatch(const MatchTriplet& match,
                              const FeatureDescriptor& feature1,
                              const FeatureDescriptor& feature2,
                              const cv::Mat& motion_status1,
                              const cv::Mat& motion_status2,
                              const std::vector<MatchTriplet>& context_matches = {});
    
    // 批量评估
    std::vector<QualityMetrics> assessBatchMatches(
        const std::vector<MatchTriplet>& matches,
        const std::vector<FeatureDescriptor>& features1,
        const std::vector<FeatureDescriptor>& features2,
        const cv::Mat& motion_status1,
        const cv::Mat& motion_status2);
    
    // 过滤低质量匹配
    std::vector<MatchTriplet> filterReliableMatches(
        const std::vector<MatchTriplet>& matches,
        const std::vector<QualityMetrics>& quality_metrics,
        float reliability_threshold = 0.6f);
    
    // 质量排序
    std::vector<size_t> rankByQuality(const std::vector<QualityMetrics>& quality_metrics);
    
    // 配置管理
    void setConfig(const QualityAssessmentConfig& config) { config_ = config; }
    const QualityAssessmentConfig& getConfig() const { return config_; }
    
    // 统计信息
    struct AssessmentStats {
        size_t total_assessed = 0;
        size_t reliable_matches = 0;
        float average_quality = 0.0f;
        std::chrono::milliseconds total_time{0};
        std::chrono::milliseconds average_time_per_match{0};
        
        float reliability_ratio() const {
            return total_assessed > 0 ? static_cast<float>(reliable_matches) / total_assessed : 0.0f;
        }
    };
    
    AssessmentStats getStats() const { return stats_; }
    void resetStats();
    
    // 机器学习支持
    struct MLTrainingData {
        std::vector<std::vector<float>> feature_vectors;
        std::vector<float> quality_labels;
        std::vector<bool> reliability_labels;
    };
    
    MLTrainingData generateTrainingData(const std::vector<QualityMetrics>& metrics);
    
    // 质量预测模型 (可选)
    void trainQualityPredictor(const MLTrainingData& training_data);
    float predictQuality(const std::vector<float>& feature_vector);
    
private:
    QualityAssessmentConfig config_;
    
    // 组件分析器
    std::unique_ptr<MotionCoherenceAnalyzer> motion_analyzer_;
    std::unique_ptr<TemporalConsistencyAnalyzer> temporal_analyzer_;
    std::unique_ptr<SpatialContinuityAnalyzer> spatial_analyzer_;
    std::unique_ptr<FeatureReliabilityAssessor> feature_assessor_;
    std::unique_ptr<StatisticalQualityAnalyzer> statistical_analyzer_;
    
    // 性能统计
    mutable AssessmentStats stats_;
    
    // 机器学习模型 (简化线性回归)
    std::vector<float> quality_model_weights_;
    bool model_trained_ = false;
    
    // 内部方法
    void initializeAnalyzers();
    
    // 运动质量分析
    float analyzeMotionQuality(const MatchTriplet& match,
                              const FeatureDescriptor& feat1,
                              const FeatureDescriptor& feat2);
    
    // 时间质量分析
    float analyzeTemporalQuality(const FeatureDescriptor& feat1,
                                const FeatureDescriptor& feat2);
    
    // 空间质量分析
    float analyzeSpatialQuality(const MatchTriplet& match,
                               const cv::Mat& motion_status1,
                               const cv::Mat& motion_status2,
                               const std::vector<MatchTriplet>& context_matches);
    
    // 特征质量分析
    float analyzeFeatureQuality(const FeatureDescriptor& feat1,
                               const FeatureDescriptor& feat2);
    
    // 统计质量分析
    float analyzeStatisticalQuality(const FeatureDescriptor& feat1,
                                   const FeatureDescriptor& feat2);
    
    // 计算综合质量分数
    float computeOverallQuality(const QualityMetrics& metrics);
    
    // 并行处理支持
    void processInParallel(const std::vector<MatchTriplet>& matches,
                          const std::vector<FeatureDescriptor>& features1,
                          const std::vector<FeatureDescriptor>& features2,
                          std::vector<QualityMetrics>& results);
};

// 质量优化建议生成器
class QualityOptimizationAdvisor {
public:
    enum class IssueType {
        LOW_MOTION_COHERENCE,
        POOR_TEMPORAL_CONSISTENCY,
        WEAK_SPATIAL_CONTINUITY,
        UNRELIABLE_FEATURES,
        HIGH_NOISE_LEVEL,
        INSUFFICIENT_DATA
    };
    
    struct OptimizationSuggestion {
        IssueType issue_type;
        std::string description;
        std::string recommendation;
        float priority_score;
        std::vector<std::string> parameter_adjustments;
    };
    
    // 分析质量问题并提供建议
    std::vector<OptimizationSuggestion> analyzeAndSuggest(
        const std::vector<QualityMetrics>& quality_metrics,
        const QualityAssessmentConfig& current_config);
    
    // 生成参数调优建议
    QualityAssessmentConfig suggestParameterOptimization(
        const std::vector<QualityMetrics>& metrics,
        const QualityAssessmentConfig& current_config);
    
    // 生成质量报告
    std::string generateQualityReport(const std::vector<QualityMetrics>& metrics);
    
private:
    // 检测具体质量问题
    std::vector<IssueType> detectQualityIssues(const QualityMetrics& metrics);
    
    // 生成针对性建议
    OptimizationSuggestion generateSuggestion(IssueType issue,
                                             const QualityMetrics& metrics);
};

} // namespace VideoMatcher