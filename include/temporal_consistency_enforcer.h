#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <deque>
#include <unordered_map>
#include "video_matcher.h"
#include "feature_extractor.h"
#include "match_quality_assessor.h"

namespace VideoMatcher {

// 匹配轨迹
struct MatchTrajectory {
    int grid1_id;                                    // 第一个视频的网格ID
    int grid2_id;                                    // 第二个视频的网格ID
    std::vector<MatchTriplet> temporal_matches;      // 时间序列匹配
    std::vector<float> confidence_trajectory;        // 置信度轨迹
    std::vector<float> distance_trajectory;          // 距离轨迹
    
    // 轨迹质量评估
    float trajectory_stability = 0.0f;              // 轨迹稳定性
    float temporal_coherence = 0.0f;                // 时间连贯性
    float prediction_accuracy = 0.0f;               // 预测准确性
    float overall_quality = 0.0f;                   // 整体质量
    
    // 轨迹统计
    int trajectory_length = 0;                       // 轨迹长度
    int missing_segments = 0;                        // 缺失段数
    float average_confidence = 0.0f;                 // 平均置信度
    float confidence_variance = 0.0f;                // 置信度方差
    
    // 时间范围
    int start_time = -1;                            // 起始时间
    int end_time = -1;                              // 结束时间
    std::vector<bool> presence_mask;                 // 存在掩码
    
    // 轨迹分析
    void computeStatistics();
    float computeStability() const;
    float computeCoherence() const;
    bool isReliable(float threshold = 0.6f) const;
    
    // 轨迹预测
    MatchTriplet predictNextMatch(int time_step) const;
    float getPredictionConfidence(int time_step) const;
    
    // 轨迹修复
    void fillMissingSegments();
    void smoothTrajectory(int window_size = 5);
};

// 时间约束配置
struct TemporalConstraintConfig {
    // 基本参数
    int temporal_window_size = 16;                   // 时间窗口大小
    float smoothness_weight = 0.3f;                 // 平滑性权重
    float consistency_weight = 0.4f;                // 一致性权重
    float prediction_weight = 0.3f;                 // 预测权重
    
    // 约束参数
    float max_velocity_change = 0.2f;               // 最大速度变化
    float max_acceleration = 0.1f;                  // 最大加速度
    float min_trajectory_length = 5;                // 最小轨迹长度
    float trajectory_confidence_threshold = 0.5f;   // 轨迹置信度阈值
    
    // 平滑参数
    bool enable_temporal_smoothing = true;          // 启用时间平滑
    bool enable_trajectory_prediction = true;       // 启用轨迹预测
    bool enable_missing_interpolation = true;       // 启用缺失插值
    int smoothing_window_size = 7;                  // 平滑窗口大小
    
    // 检测参数
    float outlier_threshold = 2.0f;                 // 异常值阈值
    int min_consecutive_matches = 3;                // 最小连续匹配数
    bool enforce_monotonic_time = true;             // 强制时间单调性
    
    TemporalConstraintConfig() = default;
};

// 轨迹跟踪器
class TrajectoryTracker {
public:
    explicit TrajectoryTracker(const TemporalConstraintConfig& config = TemporalConstraintConfig{});
    
    // 主要跟踪接口
    std::vector<MatchTrajectory> trackTrajectories(
        const std::vector<std::vector<MatchTriplet>>& hierarchical_matches,
        const std::vector<std::vector<QualityMetrics>>& hierarchical_qualities);
    
    // 单层轨迹跟踪
    std::vector<MatchTrajectory> trackSingleLayerTrajectories(
        const std::vector<MatchTriplet>& matches,
        const std::vector<QualityMetrics>& qualities,
        int time_step);
    
    // 轨迹关联
    std::vector<MatchTrajectory> associateTrajectories(
        const std::vector<MatchTrajectory>& previous_trajectories,
        const std::vector<MatchTriplet>& current_matches,
        const std::vector<QualityMetrics>& current_qualities);
    
    // 轨迹初始化
    std::vector<MatchTrajectory> initializeTrajectories(
        const std::vector<MatchTriplet>& initial_matches,
        const std::vector<QualityMetrics>& initial_qualities);
    
    // 轨迹更新
    void updateTrajectory(MatchTrajectory& trajectory,
                         const MatchTriplet& new_match,
                         const QualityMetrics& quality,
                         int time_step);
    
    // 轨迹终止检测
    std::vector<MatchTrajectory> detectTerminatedTrajectories(
        const std::vector<MatchTrajectory>& active_trajectories,
        int current_time);
    
    // 配置管理
    void setConfig(const TemporalConstraintConfig& config) { config_ = config; }
    const TemporalConstraintConfig& getConfig() const { return config_; }
    
private:
    TemporalConstraintConfig config_;
    std::vector<MatchTrajectory> active_trajectories_;
    std::vector<MatchTrajectory> completed_trajectories_;
    int current_time_step_ = 0;
    
    // 轨迹关联算法
    std::vector<std::pair<int, int>> performTrajectoryAssociation(
        const std::vector<MatchTrajectory>& trajectories,
        const std::vector<MatchTriplet>& matches,
        const std::vector<QualityMetrics>& qualities);
    
    // 计算关联代价
    float computeAssociationCost(const MatchTrajectory& trajectory,
                                const MatchTriplet& match,
                                const QualityMetrics& quality);
    
    // 轨迹预测
    MatchTriplet predictTrajectoryNext(const MatchTrajectory& trajectory);
    
    // 轨迹验证
    bool validateTrajectoryUpdate(const MatchTrajectory& trajectory,
                                 const MatchTriplet& new_match);
};

// 时间平滑器
class TemporalSmoother {
public:
    enum class SmoothingMethod {
        GAUSSIAN_FILTER,    // 高斯滤波
        MEDIAN_FILTER,      // 中值滤波
        KALMAN_FILTER,      // 卡尔曼滤波
        BILATERAL_FILTER,   // 双边滤波
        SAVITZKY_GOLAY,     // Savitzky-Golay滤波
        ADAPTIVE_FILTER     // 自适应滤波
    };
    
    struct SmoothingConfig {
        SmoothingMethod method = SmoothingMethod::ADAPTIVE_FILTER;
        int window_size = 7;
        float sigma = 1.0f;
        float edge_threshold = 0.1f;
        bool preserve_edges = true;
    };
    
    TemporalSmoother(); // 默认构造函数
    explicit TemporalSmoother(const SmoothingConfig& config);
    
    // 主要平滑接口
    std::vector<MatchTriplet> smoothMatches(
        const std::vector<MatchTriplet>& raw_matches,
        const std::vector<QualityMetrics>& qualities);
    
    // 轨迹平滑
    MatchTrajectory smoothTrajectory(const MatchTrajectory& trajectory);
    
    // 距离序列平滑
    std::vector<float> smoothDistanceSequence(const std::vector<float>& distances,
                                             const std::vector<float>& weights);
    
    // 置信度序列平滑
    std::vector<float> smoothConfidenceSequence(const std::vector<float>& confidences);
    
    // 自适应平滑
    std::vector<MatchTriplet> adaptiveSmooth(
        const std::vector<MatchTriplet>& matches,
        const std::vector<QualityMetrics>& qualities);
    
private:
    SmoothingConfig config_;
    
    // 具体平滑方法实现
    std::vector<float> applyGaussianFilter(const std::vector<float>& signal, float sigma);
    std::vector<float> applyMedianFilter(const std::vector<float>& signal, int window_size);
    std::vector<float> applyKalmanFilter(const std::vector<float>& signal, 
                                        const std::vector<float>& confidence);
    std::vector<float> applyBilateralFilter(const std::vector<float>& signal, 
                                           float spatial_sigma, float intensity_sigma);
    std::vector<float> applySavitzkyGolayFilter(const std::vector<float>& signal, 
                                               int window_size, int poly_order = 3);
    
    // 边缘检测和保护
    std::vector<bool> detectTemporalEdges(const std::vector<float>& signal, float threshold);
    std::vector<float> preserveEdges(const std::vector<float>& original,
                                    const std::vector<float>& smoothed,
                                    const std::vector<bool>& edges);
};

// 异常检测器
class TemporalOutlierDetector {
public:
    enum class DetectionMethod {
        STATISTICAL,        // 统计方法
        ISOLATION_FOREST,   // 孤立森林
        LOCAL_OUTLIER,      // 局部异常因子
        TEMPORAL_PATTERN,   // 时间模式分析
        ENSEMBLE           // 集成方法
    };
    
    struct OutlierInfo {
        int time_index;
        MatchTriplet outlier_match;
        float outlier_score;
        std::string detection_reason;
        std::vector<std::string> correction_suggestions;
    };
    
    explicit TemporalOutlierDetector(DetectionMethod method = DetectionMethod::ENSEMBLE);
    
    // 检测时间异常
    std::vector<OutlierInfo> detectOutliers(
        const std::vector<MatchTriplet>& matches,
        const std::vector<QualityMetrics>& qualities);
    
    // 检测轨迹异常
    std::vector<OutlierInfo> detectTrajectoryOutliers(
        const MatchTrajectory& trajectory);
    
    // 统计异常检测
    std::vector<OutlierInfo> detectStatisticalOutliers(
        const std::vector<float>& values,
        float z_threshold = 2.5f);
    
    // 时间模式异常检测
    std::vector<OutlierInfo> detectPatternOutliers(
        const std::vector<MatchTriplet>& matches,
        int pattern_window = 5);
    
    // 异常修正
    std::vector<MatchTriplet> correctOutliers(
        const std::vector<MatchTriplet>& matches,
        const std::vector<OutlierInfo>& outliers);
    
private:
    DetectionMethod method_;
    
    // 计算异常分数
    float computeOutlierScore(const MatchTriplet& match,
                             const std::vector<MatchTriplet>& context,
                             const std::vector<QualityMetrics>& qualities);
    
    // 局部异常因子
    float computeLocalOutlierFactor(const MatchTriplet& match,
                                   const std::vector<MatchTriplet>& neighbors);
    
    // 时间模式分析
    float analyzeTemporalPattern(const std::vector<MatchTriplet>& window);
};

// 缺失数据插值器
class MissingDataInterpolator {
public:
    enum class InterpolationMethod {
        LINEAR,             // 线性插值
        CUBIC_SPLINE,       // 三次样条
        POLYNOMIAL,         // 多项式插值
        KALMAN_PREDICTION,  // 卡尔曼预测
        PATTERN_BASED,      // 基于模式的插值
        ML_PREDICTION       // 机器学习预测
    };
    
    struct InterpolationConfig {
        InterpolationMethod method = InterpolationMethod::PATTERN_BASED;
        int max_gap_size = 5;           // 最大缺失段长度
        float confidence_decay = 0.8f;   // 置信度衰减
        bool validate_interpolation = true; // 验证插值结果
    };
    
    MissingDataInterpolator(); // 默认构造函数
    explicit MissingDataInterpolator(const InterpolationConfig& config);
    
    // 插值缺失匹配
    std::vector<MatchTriplet> interpolateMissingMatches(
        const std::vector<MatchTriplet>& matches,
        const std::vector<bool>& presence_mask,
        const std::vector<QualityMetrics>& qualities);
    
    // 插值轨迹缺失段
    MatchTrajectory interpolateTrajectoryGaps(const MatchTrajectory& trajectory);
    
    // 线性插值
    MatchTriplet linearInterpolate(const MatchTriplet& match1,
                                  const MatchTriplet& match2,
                                  float ratio);
    
    // 基于模式的插值
    MatchTriplet patternBasedInterpolate(const std::vector<MatchTriplet>& context,
                                        int missing_index);
    
    // 卡尔曼预测插值
    MatchTriplet kalmanInterpolate(const std::vector<MatchTriplet>& history,
                                  int steps_ahead);
    
    // 验证插值质量
    float validateInterpolation(const MatchTriplet& interpolated,
                               const std::vector<MatchTriplet>& context);
    
private:
    InterpolationConfig config_;
    
    // 检测缺失段
    std::vector<std::pair<int, int>> detectMissingSegments(const std::vector<bool>& presence_mask);
    
    // 多项式拟合
    std::vector<float> polynomialFit(const std::vector<float>& x,
                                    const std::vector<float>& y,
                                    int degree);
    
    // 三次样条插值
    std::vector<float> cubicSplineInterpolate(const std::vector<float>& x,
                                             const std::vector<float>& y,
                                             const std::vector<float>& xi);
};

// 预测器
class MatchPredictor {
public:
    struct PredictionModel {
        std::vector<float> position_trend;      // 位置趋势
        std::vector<float> velocity_estimate;   // 速度估计
        std::vector<float> distance_trend;      // 距离趋势
        float model_confidence;                 // 模型置信度
        int prediction_horizon;                 // 预测范围
    };
    
    explicit MatchPredictor();
    
    // 训练预测模型
    PredictionModel trainModel(const MatchTrajectory& trajectory);
    
    // 预测下一个匹配
    MatchTriplet predictNext(const PredictionModel& model,
                            const MatchTrajectory& trajectory);
    
    // 多步预测
    std::vector<MatchTriplet> predictMultiStep(const PredictionModel& model,
                                              const MatchTrajectory& trajectory,
                                              int steps);
    
    // 预测置信度评估
    float assessPredictionConfidence(const PredictionModel& model,
                                    const MatchTrajectory& trajectory,
                                    int prediction_step);
    
    // 更新预测模型
    void updateModel(PredictionModel& model, const MatchTriplet& actual_match);
    
private:
    // 运动模型
    struct MotionModel {
        cv::Point2f position;
        cv::Point2f velocity;
        cv::Point2f acceleration;
        float distance;
        float distance_velocity;
    };
    
    MotionModel estimateMotionModel(const MatchTrajectory& trajectory);
    MatchTriplet predictFromMotionModel(const MotionModel& model, int steps);
    
    // 卡尔曼滤波器
    class KalmanPredictor {
    public:
        KalmanPredictor();
        void initialize(const MatchTriplet& initial_match);
        MatchTriplet predict();
        void update(const MatchTriplet& measurement);
        float getConfidence() const { return confidence_; }
        
    private:
        cv::Mat state_;           // 状态向量 [x, y, dx, dy, distance]
        cv::Mat covariance_;      // 协方差矩阵
        cv::Mat transition_;      // 状态转移矩阵
        cv::Mat measurement_matrix_; // 测量矩阵
        cv::Mat process_noise_;   // 过程噪声
        cv::Mat measurement_noise_; // 测量噪声
        float confidence_ = 0.0f;
        bool initialized_ = false;
    };
    
    std::unique_ptr<KalmanPredictor> kalman_predictor_;
};

// 时间一致性强制器主类
class TemporalConsistencyEnforcer {
public:
    explicit TemporalConsistencyEnforcer(const TemporalConstraintConfig& config = TemporalConstraintConfig{});
    
    // 主要一致性强制接口
    std::vector<MatchTriplet> enforceTemporalConsistency(
        const std::vector<MatchTriplet>& raw_matches,
        const std::vector<QualityMetrics>& qualities,
        const cv::Mat& motion_status1,
        const cv::Mat& motion_status2,
        int temporal_window = 16);
    
    // 分层时间一致性强制
    std::vector<std::vector<MatchTriplet>> enforceHierarchicalConsistency(
        const std::vector<std::vector<MatchTriplet>>& hierarchical_matches,
        const std::vector<std::vector<QualityMetrics>>& hierarchical_qualities,
        const std::vector<cv::Mat>& motion_status_pyramid1,
        const std::vector<cv::Mat>& motion_status_pyramid2);
    
    // 轨迹跟踪和管理
    std::vector<MatchTrajectory> trackMatchTrajectories(
        const std::vector<std::vector<MatchTriplet>>& hierarchical_matches,
        const std::vector<std::vector<QualityMetrics>>& hierarchical_qualities);
    
    // 时间平滑
    std::vector<MatchTriplet> smoothTemporalMatches(
        const std::vector<MatchTriplet>& matches,
        const std::vector<QualityMetrics>& qualities);
    
    // 异常检测和修正
    std::vector<MatchTriplet> detectAndCorrectOutliers(
        const std::vector<MatchTriplet>& matches,
        const std::vector<QualityMetrics>& qualities);
    
    // 缺失数据插值
    std::vector<MatchTriplet> interpolateMissingData(
        const std::vector<MatchTriplet>& matches,
        const std::vector<bool>& presence_mask,
        const std::vector<QualityMetrics>& qualities);
    
    // 配置管理
    void setConfig(const TemporalConstraintConfig& config) { config_ = config; }
    const TemporalConstraintConfig& getConfig() const { return config_; }
    
    // 性能统计
    struct ConsistencyStats {
        size_t total_matches_processed = 0;
        size_t outliers_detected = 0;
        size_t outliers_corrected = 0;
        size_t missing_interpolated = 0;
        size_t trajectories_tracked = 0;
        float average_trajectory_quality = 0.0f;
        std::chrono::milliseconds processing_time{0};
    };
    
    ConsistencyStats getStats() const { return stats_; }
    void resetStats();
    
    // 质量评估
    float assessTemporalQuality(const std::vector<MatchTriplet>& matches,
                               const std::vector<QualityMetrics>& qualities);
    
    float assessTrajectoryQuality(const MatchTrajectory& trajectory);
    
private:
    TemporalConstraintConfig config_;
    
    // 核心组件
    std::unique_ptr<TrajectoryTracker> trajectory_tracker_;
    std::unique_ptr<TemporalSmoother> temporal_smoother_;
    std::unique_ptr<TemporalOutlierDetector> outlier_detector_;
    std::unique_ptr<MissingDataInterpolator> interpolator_;
    std::unique_ptr<MatchPredictor> predictor_;
    
    // 性能统计
    mutable ConsistencyStats stats_;
    
    // 内部方法
    void initializeComponents();
    
    // 时间窗口处理
    std::vector<MatchTriplet> processTemporalWindow(
        const std::vector<MatchTriplet>& window_matches,
        const std::vector<QualityMetrics>& window_qualities,
        int window_start, int window_size);
    
    // 约束验证
    bool validateTemporalConstraints(const std::vector<MatchTriplet>& matches);
    
    // 一致性评分
    float computeConsistencyScore(const std::vector<MatchTriplet>& matches,
                                 const std::vector<QualityMetrics>& qualities);
    
    // 自适应参数调整
    void adaptParameters(const ConsistencyStats& current_stats);
};

// 时间对齐优化器
class TemporalAlignmentOptimizer {
public:
    struct AlignmentResult {
        std::vector<MatchTriplet> aligned_matches;
        std::vector<int> time_offsets;
        float alignment_quality;
        float temporal_consistency;
    };
    
    // 优化时间对齐
    AlignmentResult optimizeTemporalAlignment(
        const std::vector<MatchTriplet>& matches1,
        const std::vector<MatchTriplet>& matches2,
        const std::vector<QualityMetrics>& qualities1,
        const std::vector<QualityMetrics>& qualities2);
    
    // 局部时间对齐
    AlignmentResult optimizeLocalAlignment(
        const std::vector<MatchTriplet>& matches,
        const std::vector<QualityMetrics>& qualities,
        int alignment_window = 10);
    
private:
    // 计算最优时间偏移
    int computeOptimalOffset(const std::vector<MatchTriplet>& seq1,
                            const std::vector<MatchTriplet>& seq2,
                            int max_offset = 10);
    
    // 动态时间规整
    AlignmentResult performDynamicTimeWarping(
        const std::vector<MatchTriplet>& seq1,
        const std::vector<MatchTriplet>& seq2);
};

} // namespace VideoMatcher