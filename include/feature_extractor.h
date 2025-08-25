#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <memory>
#include <string>

namespace VideoMatcher {

// 多尺度特征描述符
struct FeatureDescriptor {
    // 运动特征
    std::vector<float> motion_magnitude;    // 运动幅度
    std::vector<float> motion_direction;    // 运动方向
    std::vector<float> motion_consistency;  // 运动一致性
    
    // 纹理特征  
    std::vector<float> lbp_features;        // LBP纹理特征
    std::vector<float> gradient_features;   // 梯度特征
    std::vector<float> intensity_stats;     // 亮度统计特征
    
    // 时序特征
    std::vector<float> temporal_patterns;   // 时间模式
    std::vector<float> frequency_features;  // 频域特征
    std::vector<float> autocorr_features;   // 自相关特征
    
    // 几何特征
    std::vector<float> spatial_moments;     // 空间矩特征
    std::vector<float> shape_features;      // 形状特征
    
    // 质量评估
    float confidence_score = 0.0f;         // 特征质量分数
    float motion_reliability = 0.0f;       // 运动可靠性
    float texture_richness = 0.0f;         // 纹理丰富度
    
    // 构造函数
    FeatureDescriptor() = default;
    
    // 特征维度
    size_t getTotalDimension() const;
    
    // 特征归一化
    void normalize();
    
    // 特征融合权重
    struct WeightConfig {
        float motion_weight = 0.4f;
        float texture_weight = 0.3f; 
        float temporal_weight = 0.2f;
        float geometric_weight = 0.1f;
    };
    
    // 计算加权特征向量
    std::vector<float> getWeightedFeatureVector(const WeightConfig& weights) const;
};

// 光流计算器
class OpticalFlowCalculator {
public:
    struct FlowResult {
        cv::Mat flow_magnitude;
        cv::Mat flow_direction;
        cv::Mat flow_confidence;
        float average_motion;
        float motion_variance;
    };
    
    // 构造函数
    explicit OpticalFlowCalculator(const cv::String& method = "farneback");
    
    // 计算光流
    FlowResult computeOpticalFlow(const cv::Mat& frame1, const cv::Mat& frame2);
    
    // 计算网格区域光流
    FlowResult computeGridFlow(const cv::Mat& frame1, const cv::Mat& frame2, 
                              const cv::Rect& grid_rect);
    
    // 光流质量评估
    float assessFlowQuality(const FlowResult& flow);
    
private:
    cv::String method_;
    cv::Ptr<cv::FarnebackOpticalFlow> farneback_;
    // cv::Ptr<cv::cuda::OpticalFlowDual_TVL1> tvl1_gpu_;  // GPU版本 - 已废弃，使用新的接口
    bool use_gpu_ = false;
};

// 纹理特征提取器
class TextureFeatureExtractor {
public:
    struct TextureFeatures {
        std::vector<float> lbp_histogram;      // LBP直方图
        std::vector<float> glcm_features;      // 灰度共生矩阵特征
        std::vector<float> gabor_responses;    // Gabor滤波响应
        std::vector<float> wavelet_features;   // 小波特征
        std::vector<float> intensity_stats;    // 亮度统计特征
        float texture_energy;                  // 纹理能量
        float texture_contrast;                // 纹理对比度
        float texture_homogeneity;             // 纹理均匀性
    };
    
    // 构造函数
    TextureFeatureExtractor();
    
    // 提取纹理特征
    TextureFeatures extractTextureFeatures(const cv::Mat& image, 
                                          const cv::Rect& region);
    
    // LBP特征
    cv::Mat computeLBP(const cv::Mat& image, int radius = 1, int neighbors = 8);
    
    // Gabor滤波器组
    std::vector<cv::Mat> applyGaborFilters(const cv::Mat& image);
    
    // 纹理质量评估
    float assessTextureQuality(const TextureFeatures& features);
    
private:
    std::vector<cv::Mat> gabor_kernels_;
    void initializeGaborKernels();
};

// 时序特征分析器
class TemporalFeatureAnalyzer {
public:
    struct TemporalFeatures {
        std::vector<float> autocorrelation;    // 自相关函数
        std::vector<float> fft_magnitude;      // FFT幅度谱
        std::vector<float> fft_phase;          // FFT相位谱
        std::vector<float> trend_features;     // 趋势特征
        std::vector<float> periodicity;        // 周期性特征
        float temporal_stability;              // 时间稳定性
        float dominant_frequency;              // 主频率
        float spectral_entropy;                // 谱熵
    };
    
    // 构造函数
    explicit TemporalFeatureAnalyzer(int window_size = 32);
    
    // 提取时序特征
    TemporalFeatures extractTemporalFeatures(const std::vector<float>& time_series);
    
    // 运动序列分析
    TemporalFeatures analyzeMotionSequence(const std::vector<bool>& motion_sequence);
    
    // 时序模式匹配
    float computeTemporalSimilarity(const TemporalFeatures& feat1, 
                                   const TemporalFeatures& feat2);
    
    // 时序质量评估
    float assessTemporalQuality(const TemporalFeatures& features);
    
private:
    int window_size_;
    cv::Mat fft_buffer_;
    
    // 计算自相关
    std::vector<float> computeAutocorrelation(const std::vector<float>& signal, int max_lag);
    
    // FFT分析
    void performFFTAnalysis(const std::vector<float>& signal, 
                           std::vector<float>& magnitude,
                           std::vector<float>& phase);
};

// 几何特征计算器
class GeometricFeatureCalculator {
public:
    struct GeometricFeatures {
        std::vector<float> hu_moments;         // Hu不变矩
        std::vector<float> zernike_moments;    // Zernike矩
        std::vector<float> shape_descriptors;  // 形状描述符
        cv::Point2f centroid;                  // 质心
        float area;                            // 面积
        float perimeter;                       // 周长
        float eccentricity;                    // 偏心率
        float compactness;                     // 紧致度
    };
    
    // 构造函数
    GeometricFeatureCalculator();
    
    // 提取几何特征
    GeometricFeatures extractGeometricFeatures(const cv::Mat& binary_mask);
    
    // 计算形状相似性
    float computeShapeSimilarity(const GeometricFeatures& feat1,
                                const GeometricFeatures& feat2);
    
    // 几何质量评估
    float assessGeometricQuality(const GeometricFeatures& features);
    
private:
    // 计算Zernike矩
    std::vector<float> computeZernikeMoments(const cv::Mat& image, int max_order);
};

// 多尺度特征提取器主类
class MultiScaleFeatureExtractor {
public:
    struct ExtractionConfig {
        bool enable_motion_features = true;
        bool enable_texture_features = true;
        bool enable_temporal_features = true;
        bool enable_geometric_features = true;
        bool use_gpu_acceleration = false;
        int temporal_window_size = 32;
        float quality_threshold = 0.3f;
    };
    
    // 构造函数
    MultiScaleFeatureExtractor(); // 默认构造函数
    explicit MultiScaleFeatureExtractor(const ExtractionConfig& config);
    
    // 主要特征提取接口
    FeatureDescriptor extractFeatures(const std::vector<cv::Mat>& frame_sequence,
                                     const cv::Rect& grid_rect,
                                     const std::vector<bool>& motion_sequence);
    
    // 批量特征提取
    std::vector<FeatureDescriptor> extractBatchFeatures(
        const std::vector<cv::Mat>& frame_sequence,
        const std::vector<cv::Rect>& grid_rects,
        const cv::Mat& motion_status_matrix);
    
    // 特征质量评估
    float assessOverallQuality(const FeatureDescriptor& descriptor);
    
    // 配置管理
    void setConfig(const ExtractionConfig& config) { config_ = config; }
    const ExtractionConfig& getConfig() const { return config_; }
    
    // 性能监控
    struct PerformanceStats {
        std::chrono::milliseconds total_time;
        std::chrono::milliseconds motion_time;
        std::chrono::milliseconds texture_time;
        std::chrono::milliseconds temporal_time;
        std::chrono::milliseconds geometric_time;
        size_t features_extracted;
        float average_quality;
    };
    
    PerformanceStats getPerformanceStats() const { return perf_stats_; }
    void resetPerformanceStats();
    
private:
    ExtractionConfig config_;
    
    // 特征提取器组件
    std::unique_ptr<OpticalFlowCalculator> flow_calculator_;
    std::unique_ptr<TextureFeatureExtractor> texture_extractor_;
    std::unique_ptr<TemporalFeatureAnalyzer> temporal_analyzer_;
    std::unique_ptr<GeometricFeatureCalculator> geometric_calculator_;
    
    // 性能统计
    mutable PerformanceStats perf_stats_;
    mutable std::chrono::high_resolution_clock::time_point start_time_;
    
    // 内部方法
    void extractMotionFeatures(const std::vector<cv::Mat>& frames,
                              const cv::Rect& rect,
                              FeatureDescriptor& descriptor);
    
    void extractTextureFeatures(const cv::Mat& frame,
                               const cv::Rect& rect,
                               FeatureDescriptor& descriptor);
    
    void extractTemporalFeatures(const std::vector<bool>& motion_sequence,
                                FeatureDescriptor& descriptor);
    
    void extractGeometricFeatures(const cv::Mat& motion_mask,
                                 FeatureDescriptor& descriptor);
    
    // 质量评估内部方法
    float computeFeatureReliability(const FeatureDescriptor& descriptor);
    
    // GPU内存管理
    std::vector<cv::cuda::GpuMat> gpu_buffers_;
    bool gpu_initialized_ = false;
    void initializeGPU();
};

// 特征距离计算器
class FeatureDistanceCalculator {
public:
    enum class DistanceMetric {
        EUCLIDEAN,          // 欧氏距离
        COSINE,             // 余弦距离
        MANHATTAN,          // 曼哈顿距离
        CHEBYSHEV,          // 切比雪夫距离
        CORRELATION,        // 相关距离
        CHI_SQUARE,         // 卡方距离
        BHATTACHARYYA,      // 巴氏距离
        WEIGHTED_FUSION     // 加权融合距离
    };
    
    // 构造函数
    explicit FeatureDistanceCalculator(DistanceMetric default_metric = DistanceMetric::WEIGHTED_FUSION);
    
    // 计算特征距离
    float computeDistance(const FeatureDescriptor& feat1,
                         const FeatureDescriptor& feat2,
                         DistanceMetric metric = DistanceMetric::WEIGHTED_FUSION);
    
    // 批量距离计算
    std::vector<std::vector<float>> computeBatchDistances(
        const std::vector<FeatureDescriptor>& features1,
        const std::vector<FeatureDescriptor>& features2,
        DistanceMetric metric = DistanceMetric::WEIGHTED_FUSION);
    
    // 设置距离权重
    void setDistanceWeights(const FeatureDescriptor::WeightConfig& weights);
    
    // GPU加速批量计算
    void computeDistancesGPU(const float* features1, const float* features2,
                            float* distances, int n1, int n2, int feature_dim);
    
private:
    DistanceMetric default_metric_;
    FeatureDescriptor::WeightConfig distance_weights_;
    
    // 具体距离计算方法
    float computeEuclideanDistance(const std::vector<float>& f1, const std::vector<float>& f2);
    float computeCosineDistance(const std::vector<float>& f1, const std::vector<float>& f2);
    float computeManhattanDistance(const std::vector<float>& f1, const std::vector<float>& f2);
    float computeCorrelationDistance(const std::vector<float>& f1, const std::vector<float>& f2);
    float computeChiSquareDistance(const std::vector<float>& f1, const std::vector<float>& f2);
    float computeBhattacharyyaDistance(const std::vector<float>& f1, const std::vector<float>& f2);
    
    // 加权融合距离
    float computeWeightedFusionDistance(const FeatureDescriptor& feat1,
                                       const FeatureDescriptor& feat2);
};

} // namespace VideoMatcher