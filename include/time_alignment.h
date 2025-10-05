#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <unordered_map>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace VideoMatcher {


struct LocalRegionFeature {
    std::vector<float> activation_rate;   
    std::vector<float> change_rate;       
    std::vector<float> persistence;         
    float quality_score = 0.0f;            
    
    LocalRegionFeature() = default;
    explicit LocalRegionFeature(size_t seq_len) {
        activation_rate.reserve(seq_len);
        change_rate.reserve(seq_len);
        persistence.reserve(seq_len);
    }
};

struct RegionMatchPair {
    int region1_id;
    int region2_id;
    float similarity_score;
    float confidence;
    int detected_offset;
    
    RegionMatchPair(int r1, int r2, float sim, float conf, int offset)
        : region1_id(r1), region2_id(r2), similarity_score(sim), 
          confidence(conf), detected_offset(offset) {}
};


struct TimeAlignmentResult {
    int detected_offset = 0;           
    float confidence = 0.0f;           
    int num_reliable_regions = 0;      
    std::vector<RegionMatchPair> region_pairs;  
    bool is_valid = false;             
    
    TimeAlignmentResult() = default;
};

struct TimeAlignmentParameters {
    bool enable_time_alignment = false;     // 是否启用时间对齐
    int max_time_offset = 30;               // 最大搜索偏移范围
    int region_grid_size = 4;               // 区域划分网格大小 (4x4=16个区域)
    float region_overlap_ratio = 0.5f;      // 区域重叠比例
    float similarity_threshold = 0.6f;      // 相似性阈值
    int min_reliable_regions = 3;           // 最少可靠区域对数
    int smoothing_window = 5;               // 时间平滑窗口大小
    bool use_coarse_detection = true;       // 是否使用粗检测+精检测
    int coarse_downsample_factor = 4;       // 粗检测降采样因子
    
    TimeAlignmentParameters() = default;
};

// 主时间对齐类
class TimeAlignmentEngine {
private:
    TimeAlignmentParameters params_;
    
    // 内部辅助结构
    struct GridDimensions {
        int rows, cols;
        int total_grids;
        cv::Size frame_size;
        
        GridDimensions(const cv::Mat& motion_status) {
            // 假设motion_status是网格x时间的矩阵，需要推断空间维度
            total_grids = motion_status.rows;
            // 简化：假设是正方形网格布局
            cols = static_cast<int>(std::sqrt(total_grids));
            rows = (total_grids + cols - 1) / cols;
        }
    };
    
public:
    explicit TimeAlignmentEngine(const TimeAlignmentParameters& params = TimeAlignmentParameters{})
        : params_(params) {}
    
    // 主要接口：检测两个运动状态矩阵之间的时间偏移
    TimeAlignmentResult detectTimeOffset(const cv::Mat& motion_status1, 
                                        const cv::Mat& motion_status2);
    
    // 应用时间偏移到运动状态矩阵
    static cv::Mat applyTimeOffset(const cv::Mat& motion_status, int offset);
    
    // 设置参数
    void setParameters(const TimeAlignmentParameters& params) { params_ = params; }
    const TimeAlignmentParameters& getParameters() const { return params_; }
    
private:
    // 空间分块：将网格划分为重叠的局部区域
    std::vector<std::vector<int>> createSpatialRegions(const GridDimensions& grid_dims);
    
    // 提取局部区域的运动特征
    LocalRegionFeature extractRegionFeature(const cv::Mat& motion_status, 
                                           const std::vector<int>& region_grids);
    
    // 计算两个特征序列的相似性
    float calculateFeatureSimilarity(const LocalRegionFeature& feature1, 
                                    const LocalRegionFeature& feature2);
    
    // 使用归一化互相关检测时间偏移
    int detectOffsetBetweenFeatures(const LocalRegionFeature& feature1, 
                                   const LocalRegionFeature& feature2, 
                                   float& confidence);
    
    // 归一化互相关计算
    double calculateNormalizedCrossCorrelation(const std::vector<float>& seq1, 
                                              const std::vector<float>& seq2, 
                                              int offset, int compare_length);
    
    // RANSAC鲁棒偏移估计
    int robustOffsetEstimation(const std::vector<RegionMatchPair>& region_pairs);
    
    // 时间序列降采样
    cv::Mat temporalDownsample(const cv::Mat& motion_status, int factor);
    
    // 移动平均平滑
    std::vector<float> movingAverage(const std::vector<float>& sequence, int window_size);
    
    // 计算激活率序列
    std::vector<float> calculateActivationRate(const cv::Mat& motion_status, 
                                              const std::vector<int>& region_grids);
    
    // 计算变化率序列
    std::vector<float> calculateChangeRate(const std::vector<float>& activation_rate);
    
    // 计算持续性序列
    std::vector<float> calculatePersistence(const std::vector<float>& activation_rate, int window_size);
};

} // namespace VideoMatcher 