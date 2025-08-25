#include "feature_extractor.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fftw3.h>  // 需要安装FFTW库进行FFT计算
#ifdef _OPENMP
#include <omp.h>
#endif

namespace VideoMatcher {

// FeatureDescriptor 实现
size_t FeatureDescriptor::getTotalDimension() const {
    return motion_magnitude.size() + motion_direction.size() + motion_consistency.size() +
           lbp_features.size() + gradient_features.size() + intensity_stats.size() +
           temporal_patterns.size() + frequency_features.size() + autocorr_features.size() +
           spatial_moments.size() + shape_features.size();
}

void FeatureDescriptor::normalize() {
    auto normalizeVector = [](std::vector<float>& vec) {
        if (vec.empty()) return;
        float mean = std::accumulate(vec.begin(), vec.end(), 0.0f) / vec.size();
        float variance = 0.0f;
        for (float val : vec) {
            variance += (val - mean) * (val - mean);
        }
        variance /= vec.size();
        float std_dev = std::sqrt(variance + 1e-8f);
        
        for (float& val : vec) {
            val = (val - mean) / std_dev;
        }
    };
    
    normalizeVector(motion_magnitude);
    normalizeVector(motion_direction);
    normalizeVector(motion_consistency);
    normalizeVector(lbp_features);
    normalizeVector(gradient_features);
    normalizeVector(intensity_stats);
    normalizeVector(temporal_patterns);
    normalizeVector(frequency_features);
    normalizeVector(autocorr_features);
    normalizeVector(spatial_moments);
    normalizeVector(shape_features);
}

std::vector<float> FeatureDescriptor::getWeightedFeatureVector(const WeightConfig& weights) const {
    std::vector<float> weighted_features;
    
    // 运动特征
    for (float val : motion_magnitude) weighted_features.push_back(val * weights.motion_weight);
    for (float val : motion_direction) weighted_features.push_back(val * weights.motion_weight);
    for (float val : motion_consistency) weighted_features.push_back(val * weights.motion_weight);
    
    // 纹理特征
    for (float val : lbp_features) weighted_features.push_back(val * weights.texture_weight);
    for (float val : gradient_features) weighted_features.push_back(val * weights.texture_weight);
    for (float val : intensity_stats) weighted_features.push_back(val * weights.texture_weight);
    
    // 时序特征
    for (float val : temporal_patterns) weighted_features.push_back(val * weights.temporal_weight);
    for (float val : frequency_features) weighted_features.push_back(val * weights.temporal_weight);
    for (float val : autocorr_features) weighted_features.push_back(val * weights.temporal_weight);
    
    // 几何特征
    for (float val : spatial_moments) weighted_features.push_back(val * weights.geometric_weight);
    for (float val : shape_features) weighted_features.push_back(val * weights.geometric_weight);
    
    return weighted_features;
}

// OpticalFlowCalculator 实现
OpticalFlowCalculator::OpticalFlowCalculator(const cv::String& method) : method_(method) {
    if (method_ == "farneback") {
        farneback_ = cv::FarnebackOpticalFlow::create();
    }
    
    // 检查GPU可用性
    try {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            // tvl1_gpu_ = cv::cuda::OpticalFlowDual_TVL1::create(); // 已废弃
            use_gpu_ = true;
        }
    } catch (...) {
        use_gpu_ = false;
    }
}

OpticalFlowCalculator::FlowResult OpticalFlowCalculator::computeOpticalFlow(
    const cv::Mat& frame1, const cv::Mat& frame2) {
    
    FlowResult result;
    cv::Mat flow;
    
    // 转换为灰度图
    cv::Mat gray1, gray2;
    if (frame1.channels() == 3) {
        cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);
    } else {
        gray1 = frame1;
        gray2 = frame2;
    }
    
    if (use_gpu_ && method_ == "tvl1") {
        // GPU加速光流计算 - 使用新的接口
        try {
            cv::cuda::GpuMat gpu_frame1, gpu_frame2, gpu_flow;
            gpu_frame1.upload(gray1);
            gpu_frame2.upload(gray2);
            
            // 使用OpenCV CUDA光流接口
            auto farneback = cv::cuda::FarnebackOpticalFlow::create();
            farneback->calc(gpu_frame1, gpu_frame2, gpu_flow);
            gpu_flow.download(flow);
        } catch (const cv::Exception& e) {
            // GPU计算失败，回退到CPU
            cv::calcOpticalFlowFarneback(gray1, gray2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        }
    } else {
        // CPU Farneback光流
        cv::calcOpticalFlowFarneback(gray1, gray2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    }
    
    // 计算幅度和方向
    std::vector<cv::Mat> flow_parts(2);
    cv::split(flow, flow_parts);
    
    cv::cartToPolar(flow_parts[0], flow_parts[1], 
                   result.flow_magnitude, result.flow_direction, true);
    
    // 计算置信度 (基于幅度和一致性)
    cv::Mat mag_norm;
    cv::normalize(result.flow_magnitude, mag_norm, 0, 1, cv::NORM_MINMAX);
    
    // 局部一致性检查
    cv::Mat consistency;
    cv::Sobel(result.flow_direction, consistency, CV_32F, 1, 1, 3);
    cv::normalize(consistency, consistency, 0, 1, cv::NORM_MINMAX);
    result.flow_confidence = mag_norm.mul(1.0 - consistency);
    
    // 计算统计信息
    cv::Scalar mean_mag, std_mag;
    cv::meanStdDev(result.flow_magnitude, mean_mag, std_mag);
    result.average_motion = static_cast<float>(mean_mag[0]);
    result.motion_variance = static_cast<float>(std_mag[0] * std_mag[0]);
    
    return result;
}

OpticalFlowCalculator::FlowResult OpticalFlowCalculator::computeGridFlow(
    const cv::Mat& frame1, const cv::Mat& frame2, const cv::Rect& grid_rect) {
    
    // 确保网格区域在图像范围内
    cv::Rect safe_rect = grid_rect & cv::Rect(0, 0, frame1.cols, frame1.rows);
    
    cv::Mat grid1 = frame1(safe_rect);
    cv::Mat grid2 = frame2(safe_rect);
    
    return computeOpticalFlow(grid1, grid2);
}

float OpticalFlowCalculator::assessFlowQuality(const FlowResult& flow) {
    if (flow.flow_magnitude.empty()) return 0.0f;
    
    // 基于多个指标评估光流质量
    float magnitude_score = std::min(1.0f, flow.average_motion / 10.0f);
    float variance_score = std::min(1.0f, flow.motion_variance / 100.0f);
    
    cv::Scalar mean_conf;
    cv::meanStdDev(flow.flow_confidence, mean_conf, cv::Scalar());
    float confidence_score = static_cast<float>(mean_conf[0]);
    
    return 0.4f * magnitude_score + 0.3f * variance_score + 0.3f * confidence_score;
}

// TextureFeatureExtractor 实现
TextureFeatureExtractor::TextureFeatureExtractor() {
    initializeGaborKernels();
}

void TextureFeatureExtractor::initializeGaborKernels() {
    // 创建Gabor滤波器组 (4个方向，3个尺度)
    gabor_kernels_.clear();
    std::vector<double> orientations = {0, CV_PI/4, CV_PI/2, 3*CV_PI/4};
    std::vector<double> scales = {0.05, 0.15, 0.25};
    
    for (double orientation : orientations) {
        for (double scale : scales) {
            cv::Mat kernel = cv::getGaborKernel(cv::Size(31, 31), 5, orientation, 
                                              2*CV_PI*scale, 0.5, 0, CV_32F);
            gabor_kernels_.push_back(kernel);
        }
    }
}

TextureFeatureExtractor::TextureFeatures TextureFeatureExtractor::extractTextureFeatures(
    const cv::Mat& image, const cv::Rect& region) {
    
    TextureFeatures features;
    
    // 确保区域在图像范围内
    cv::Rect safe_region = region & cv::Rect(0, 0, image.cols, image.rows);
    cv::Mat roi = image(safe_region);
    
    // 转换为灰度图
    cv::Mat gray;
    if (roi.channels() == 3) {
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = roi;
    }
    
    // LBP特征
    cv::Mat lbp = computeLBP(gray);
    cv::Mat lbp_hist;
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::calcHist(&lbp, 1, 0, cv::Mat(), lbp_hist, 1, &histSize, &histRange);
    cv::normalize(lbp_hist, lbp_hist, 0, 1, cv::NORM_L1);
    
    features.lbp_histogram.clear();
    for (int i = 0; i < histSize; ++i) {
        features.lbp_histogram.push_back(lbp_hist.at<float>(i));
    }
    
    // Gabor特征
    std::vector<cv::Mat> gabor_responses = applyGaborFilters(gray);
    features.gabor_responses.clear();
    for (const auto& response : gabor_responses) {
        cv::Scalar mean, std;
        cv::meanStdDev(response, mean, std);
        features.gabor_responses.push_back(static_cast<float>(mean[0]));
        features.gabor_responses.push_back(static_cast<float>(std[0]));
    }
    
    // 梯度特征
    cv::Mat grad_x, grad_y, grad_mag;
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, grad_mag);
    
    cv::Scalar grad_mean, grad_std;
    cv::meanStdDev(grad_mag, grad_mean, grad_std);
    features.texture_energy = static_cast<float>(grad_mean[0]);
    features.texture_contrast = static_cast<float>(grad_std[0]);
    
    // 统计特征
    cv::Scalar intensity_mean, intensity_std;
    cv::meanStdDev(gray, intensity_mean, intensity_std);
    features.intensity_stats.push_back(static_cast<float>(intensity_mean[0]));
    features.intensity_stats.push_back(static_cast<float>(intensity_std[0]));
    
    // 计算纹理均匀性
    cv::Mat gray_norm;
    gray.convertTo(gray_norm, CV_32F, 1.0/255.0);
    cv::Mat variance;
    cv::Laplacian(gray_norm, variance, CV_32F);
    cv::Scalar var_mean;
    cv::meanStdDev(variance, var_mean, cv::Scalar());
    features.texture_homogeneity = 1.0f / (1.0f + static_cast<float>(var_mean[0]));
    
    return features;
}

cv::Mat TextureFeatureExtractor::computeLBP(const cv::Mat& image, int radius, int neighbors) {
    cv::Mat lbp = cv::Mat::zeros(image.size(), CV_8UC1);
    
    for (int y = radius; y < image.rows - radius; ++y) {
        for (int x = radius; x < image.cols - radius; ++x) {
            uchar center = image.at<uchar>(y, x);
            uchar code = 0;
            
            for (int n = 0; n < neighbors; ++n) {
                double angle = 2.0 * CV_PI * n / neighbors;
                int nx = static_cast<int>(x + radius * cos(angle) + 0.5);
                int ny = static_cast<int>(y - radius * sin(angle) + 0.5);
                
                if (nx >= 0 && nx < image.cols && ny >= 0 && ny < image.rows) {
                    if (image.at<uchar>(ny, nx) >= center) {
                        code |= (1 << n);
                    }
                }
            }
            lbp.at<uchar>(y, x) = code;
        }
    }
    
    return lbp;
}

std::vector<cv::Mat> TextureFeatureExtractor::applyGaborFilters(const cv::Mat& image) {
    std::vector<cv::Mat> responses;
    responses.reserve(gabor_kernels_.size());
    
    for (const auto& kernel : gabor_kernels_) {
        cv::Mat response;
        cv::filter2D(image, response, CV_32F, kernel);
        responses.push_back(response);
    }
    
    return responses;
}

float TextureFeatureExtractor::assessTextureQuality(const TextureFeatures& features) {
    // 基于纹理特征的丰富度评估质量
    float energy_score = std::min(1.0f, features.texture_energy / 100.0f);
    float contrast_score = std::min(1.0f, features.texture_contrast / 50.0f);
    float homogeneity_score = features.texture_homogeneity;
    
    // LBP特征的熵
    float lbp_entropy = 0.0f;
    for (float prob : features.lbp_histogram) {
        if (prob > 0) {
            lbp_entropy -= prob * log2(prob);
        }
    }
    float entropy_score = std::min(1.0f, lbp_entropy / 8.0f);
    
    return 0.3f * energy_score + 0.3f * contrast_score + 
           0.2f * homogeneity_score + 0.2f * entropy_score;
}

// TemporalFeatureAnalyzer 实现
TemporalFeatureAnalyzer::TemporalFeatureAnalyzer(int window_size) 
    : window_size_(window_size) {
    fft_buffer_ = cv::Mat::zeros(1, window_size_, CV_32FC2);
}

TemporalFeatureAnalyzer::TemporalFeatures TemporalFeatureAnalyzer::extractTemporalFeatures(
    const std::vector<float>& time_series) {
    
    TemporalFeatures features;
    
    if (time_series.empty()) return features;
    
    // 自相关分析
    features.autocorrelation = computeAutocorrelation(time_series, std::min(20, static_cast<int>(time_series.size() / 2)));
    
    // FFT分析
    performFFTAnalysis(time_series, features.fft_magnitude, features.fft_phase);
    
    // 趋势分析
    float mean = std::accumulate(time_series.begin(), time_series.end(), 0.0f) / time_series.size();
    float trend = 0.0f;
    if (time_series.size() > 1) {
        // 简单线性趋势
        float sum_xy = 0.0f, sum_x = 0.0f, sum_x2 = 0.0f;
        for (size_t i = 0; i < time_series.size(); ++i) {
            sum_xy += i * time_series[i];
            sum_x += i;
            sum_x2 += i * i;
        }
        float n = static_cast<float>(time_series.size());
        trend = (n * sum_xy - sum_x * (mean * n)) / (n * sum_x2 - sum_x * sum_x);
    }
    features.trend_features = {mean, trend};
    
    // 计算时间稳定性
    float variance = 0.0f;
    for (float val : time_series) {
        variance += (val - mean) * (val - mean);
    }
    variance /= time_series.size();
    features.temporal_stability = 1.0f / (1.0f + variance);
    
    // 寻找主频率
    if (!features.fft_magnitude.empty()) {
        auto max_it = std::max_element(features.fft_magnitude.begin() + 1, 
                                      features.fft_magnitude.end());
        size_t max_idx = std::distance(features.fft_magnitude.begin(), max_it);
        features.dominant_frequency = static_cast<float>(max_idx) / time_series.size();
    }
    
    // 计算谱熵
    float total_power = std::accumulate(features.fft_magnitude.begin(), features.fft_magnitude.end(), 0.0f);
    if (total_power > 0) {
        features.spectral_entropy = 0.0f;
        for (float power : features.fft_magnitude) {
            if (power > 0) {
                float prob = power / total_power;
                features.spectral_entropy -= prob * log2(prob);
            }
        }
    }
    
    return features;
}

TemporalFeatureAnalyzer::TemporalFeatures TemporalFeatureAnalyzer::analyzeMotionSequence(
    const std::vector<bool>& motion_sequence) {
    
    // 将布尔序列转换为浮点序列
    std::vector<float> float_sequence;
    float_sequence.reserve(motion_sequence.size());
    for (bool val : motion_sequence) {
        float_sequence.push_back(val ? 1.0f : 0.0f);
    }
    
    return extractTemporalFeatures(float_sequence);
}

std::vector<float> TemporalFeatureAnalyzer::computeAutocorrelation(
    const std::vector<float>& signal, int max_lag) {
    
    std::vector<float> autocorr(max_lag + 1);
    
    float mean = std::accumulate(signal.begin(), signal.end(), 0.0f) / signal.size();
    
    for (int lag = 0; lag <= max_lag; ++lag) {
        float sum = 0.0f;
        int count = 0;
        
        for (size_t i = 0; i < signal.size() - lag; ++i) {
            sum += (signal[i] - mean) * (signal[i + lag] - mean);
            count++;
        }
        
        autocorr[lag] = count > 0 ? sum / count : 0.0f;
    }
    
    // 归一化
    if (autocorr[0] > 0) {
        for (float& val : autocorr) {
            val /= autocorr[0];
        }
    }
    
    return autocorr;
}

void TemporalFeatureAnalyzer::performFFTAnalysis(const std::vector<float>& signal,
                                                std::vector<float>& magnitude,
                                                std::vector<float>& phase) {
    if (signal.empty()) return;
    
    // 使用OpenCV的DFT进行FFT
    cv::Mat input = cv::Mat::zeros(1, static_cast<int>(signal.size()), CV_32FC1);
    for (size_t i = 0; i < signal.size(); ++i) {
        input.at<float>(0, static_cast<int>(i)) = signal[i];
    }
    
    cv::Mat complex_result;
    cv::dft(input, complex_result, cv::DFT_COMPLEX_OUTPUT);
    
    // 分离幅度和相位
    std::vector<cv::Mat> channels;
    cv::split(complex_result, channels);
    
    cv::Mat mag, phase_mat;
    cv::magnitude(channels[0], channels[1], mag);
    cv::phase(channels[0], channels[1], phase_mat);
    
    magnitude.clear();
    phase.clear();
    
    for (int i = 0; i < mag.cols; ++i) {
        magnitude.push_back(mag.at<float>(0, i));
        phase.push_back(phase_mat.at<float>(0, i));
    }
}

float TemporalFeatureAnalyzer::computeTemporalSimilarity(const TemporalFeatures& feat1,
                                                        const TemporalFeatures& feat2) {
    float similarity = 0.0f;
    int components = 0;
    
    // 自相关相似性
    if (!feat1.autocorrelation.empty() && !feat2.autocorrelation.empty()) {
        size_t min_size = std::min(feat1.autocorrelation.size(), feat2.autocorrelation.size());
        float corr_sim = 0.0f;
        for (size_t i = 0; i < min_size; ++i) {
            corr_sim += feat1.autocorrelation[i] * feat2.autocorrelation[i];
        }
        similarity += corr_sim / min_size;
        components++;
    }
    
    // 频域相似性
    if (!feat1.fft_magnitude.empty() && !feat2.fft_magnitude.empty()) {
        size_t min_size = std::min(feat1.fft_magnitude.size(), feat2.fft_magnitude.size());
        float freq_sim = 0.0f;
        for (size_t i = 0; i < min_size; ++i) {
            freq_sim += std::min(feat1.fft_magnitude[i], feat2.fft_magnitude[i]);
        }
        similarity += freq_sim;
        components++;
    }
    
    // 稳定性相似性
    float stability_sim = 1.0f - std::abs(feat1.temporal_stability - feat2.temporal_stability);
    similarity += stability_sim;
    components++;
    
    return components > 0 ? similarity / components : 0.0f;
}

float TemporalFeatureAnalyzer::assessTemporalQuality(const TemporalFeatures& features) {
    float quality = 0.0f;
    
    // 稳定性评分
    quality += 0.4f * features.temporal_stability;
    
    // 谱熵评分 (更高的熵表示更丰富的频率成分)
    float entropy_score = std::min(1.0f, features.spectral_entropy / 5.0f);
    quality += 0.3f * entropy_score;
    
    // 自相关衰减评分
    if (features.autocorrelation.size() > 1) {
        float decay_rate = std::abs(features.autocorrelation[0] - features.autocorrelation.back());
        quality += 0.3f * std::min(1.0f, decay_rate);
    }
    
    return quality;
}

// MultiScaleFeatureExtractor 主类实现
MultiScaleFeatureExtractor::MultiScaleFeatureExtractor() 
    : config_{} {
    
    if (config_.enable_motion_features) {
        flow_calculator_ = std::make_unique<OpticalFlowCalculator>();
    }
    
    if (config_.enable_texture_features) {
        texture_extractor_ = std::make_unique<TextureFeatureExtractor>();
    }
    
    if (config_.enable_geometric_features) {
        geometric_calculator_ = std::make_unique<GeometricFeatureCalculator>();
    }
    
    if (config_.use_gpu_acceleration) {
        initializeGPU();
    }
    
    resetPerformanceStats();
}

MultiScaleFeatureExtractor::MultiScaleFeatureExtractor(const ExtractionConfig& config) 
    : config_(config) {
    
    if (config_.enable_motion_features) {
        flow_calculator_ = std::make_unique<OpticalFlowCalculator>();
    }
    
    if (config_.enable_texture_features) {
        texture_extractor_ = std::make_unique<TextureFeatureExtractor>();
    }
    
    if (config_.enable_temporal_features) {
        temporal_analyzer_ = std::make_unique<TemporalFeatureAnalyzer>(config_.temporal_window_size);
    }
    
    if (config_.enable_geometric_features) {
        geometric_calculator_ = std::make_unique<GeometricFeatureCalculator>();
    }
    
    if (config_.use_gpu_acceleration) {
        initializeGPU();
    }
    
    resetPerformanceStats();
}

FeatureDescriptor MultiScaleFeatureExtractor::extractFeatures(
    const std::vector<cv::Mat>& frame_sequence,
    const cv::Rect& grid_rect,
    const std::vector<bool>& motion_sequence) {
    
    start_time_ = std::chrono::high_resolution_clock::now();
    
    FeatureDescriptor descriptor;
    
    if (frame_sequence.empty()) {
        return descriptor;
    }
    
    // 运动特征提取
    if (config_.enable_motion_features && flow_calculator_) {
        extractMotionFeatures(frame_sequence, grid_rect, descriptor);
    }
    
    // 纹理特征提取  
    if (config_.enable_texture_features && texture_extractor_) {
        extractTextureFeatures(frame_sequence.back(), grid_rect, descriptor);
    }
    
    // 时序特征提取
    if (config_.enable_temporal_features && temporal_analyzer_) {
        extractTemporalFeatures(motion_sequence, descriptor);
    }
    
    // 几何特征提取
    if (config_.enable_geometric_features && geometric_calculator_) {
        // 创建运动掩码
        cv::Mat motion_mask = cv::Mat::zeros(grid_rect.size(), CV_8UC1);
        // 这里需要根据motion_sequence创建二值掩码
        extractGeometricFeatures(motion_mask, descriptor);
    }
    
    // 计算质量分数
    descriptor.confidence_score = assessOverallQuality(descriptor);
    
    // 归一化特征
    descriptor.normalize();
    
    // 更新性能统计
    auto end_time = std::chrono::high_resolution_clock::now();
    perf_stats_.total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_);
    perf_stats_.features_extracted++;
    
    return descriptor;
}

void MultiScaleFeatureExtractor::extractMotionFeatures(const std::vector<cv::Mat>& frames,
                                                       const cv::Rect& rect,
                                                       FeatureDescriptor& descriptor) {
    auto start = std::chrono::high_resolution_clock::now();
    
    if (frames.size() < 2) return;
    
    // 计算连续帧之间的光流
    for (size_t i = 1; i < frames.size(); ++i) {
        auto flow_result = flow_calculator_->computeGridFlow(frames[i-1], frames[i], rect);
        
        // 提取运动特征
        cv::Scalar mean_mag, std_mag;
        cv::meanStdDev(flow_result.flow_magnitude, mean_mag, std_mag);
        descriptor.motion_magnitude.push_back(static_cast<float>(mean_mag[0]));
        descriptor.motion_consistency.push_back(static_cast<float>(std_mag[0]));
        
        // 计算主要运动方向
        cv::Scalar mean_dir;
        cv::meanStdDev(flow_result.flow_direction, mean_dir, cv::Scalar());
        descriptor.motion_direction.push_back(static_cast<float>(mean_dir[0]));
    }
    
    // 计算运动可靠性
    if (!descriptor.motion_magnitude.empty()) {
        float avg_magnitude = std::accumulate(descriptor.motion_magnitude.begin(), 
                                            descriptor.motion_magnitude.end(), 0.0f) / 
                             descriptor.motion_magnitude.size();
        descriptor.motion_reliability = std::min(1.0f, avg_magnitude / 10.0f);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    perf_stats_.motion_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

void MultiScaleFeatureExtractor::extractTextureFeatures(const cv::Mat& frame,
                                                        const cv::Rect& rect,
                                                        FeatureDescriptor& descriptor) {
    auto start = std::chrono::high_resolution_clock::now();
    
    auto texture_features = texture_extractor_->extractTextureFeatures(frame, rect);
    
    descriptor.lbp_features = texture_features.lbp_histogram;
    descriptor.gradient_features = texture_features.gabor_responses;
    descriptor.intensity_stats = texture_features.intensity_stats;
    
    // 计算纹理丰富度
    descriptor.texture_richness = texture_extractor_->assessTextureQuality(texture_features);
    
    auto end = std::chrono::high_resolution_clock::now();
    perf_stats_.texture_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

void MultiScaleFeatureExtractor::extractTemporalFeatures(const std::vector<bool>& motion_sequence,
                                                         FeatureDescriptor& descriptor) {
    auto start = std::chrono::high_resolution_clock::now();
    
    auto temporal_features = temporal_analyzer_->analyzeMotionSequence(motion_sequence);
    
    descriptor.temporal_patterns = temporal_features.autocorrelation;
    descriptor.frequency_features = temporal_features.fft_magnitude;
    descriptor.autocorr_features = temporal_features.autocorrelation;
    
    auto end = std::chrono::high_resolution_clock::now();
    perf_stats_.temporal_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

void MultiScaleFeatureExtractor::extractGeometricFeatures(const cv::Mat& motion_mask,
                                                          FeatureDescriptor& descriptor) {
    auto start = std::chrono::high_resolution_clock::now();
    
    if (geometric_calculator_) {
        auto geometric_features = geometric_calculator_->extractGeometricFeatures(motion_mask);
        descriptor.spatial_moments = geometric_features.hu_moments;
        descriptor.shape_features = geometric_features.zernike_moments;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    perf_stats_.geometric_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
}

float MultiScaleFeatureExtractor::assessOverallQuality(const FeatureDescriptor& descriptor) {
    float total_quality = 0.0f;
    int components = 0;
    
    if (config_.enable_motion_features && descriptor.motion_reliability > 0) {
        total_quality += descriptor.motion_reliability;
        components++;
    }
    
    if (config_.enable_texture_features && descriptor.texture_richness > 0) {
        total_quality += descriptor.texture_richness;
        components++;
    }
    
    // 可以根据其他特征继续添加质量评估
    
    return components > 0 ? total_quality / components : 0.0f;
}

void MultiScaleFeatureExtractor::initializeGPU() {
    try {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            // 预分配GPU内存缓冲区
            gpu_buffers_.resize(4);
            for (auto& buffer : gpu_buffers_) {
                buffer.create(480, 640, CV_8UC1);  // 预分配典型尺寸
            }
            gpu_initialized_ = true;
        }
    } catch (...) {
        gpu_initialized_ = false;
    }
}

void MultiScaleFeatureExtractor::resetPerformanceStats() {
    perf_stats_ = PerformanceStats{};
}

// GeometricFeatureCalculator 简化实现
GeometricFeatureCalculator::GeometricFeatureCalculator() {
}

GeometricFeatureCalculator::GeometricFeatures GeometricFeatureCalculator::extractGeometricFeatures(
    const cv::Mat& binary_mask) {
    
    GeometricFeatures features;
    
    if (binary_mask.empty()) return features;
    
    // 计算Hu矩
    cv::Moments moments = cv::moments(binary_mask);
    cv::HuMoments(moments, features.hu_moments);
    
    // 计算基本几何属性
    features.area = static_cast<float>(moments.m00);
    features.centroid.x = static_cast<float>(moments.m10 / moments.m00);
    features.centroid.y = static_cast<float>(moments.m01 / moments.m00);
    
    // 寻找轮廓计算周长
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (!contours.empty()) {
        features.perimeter = static_cast<float>(cv::arcLength(contours[0], true));
        features.compactness = 4 * CV_PI * features.area / (features.perimeter * features.perimeter);
    }
    
    return features;
}

float GeometricFeatureCalculator::computeShapeSimilarity(const GeometricFeatures& feat1,
                                                        const GeometricFeatures& feat2) {
    if (feat1.hu_moments.empty() || feat2.hu_moments.empty()) return 0.0f;
    
    float similarity = 0.0f;
    for (size_t i = 0; i < std::min(feat1.hu_moments.size(), feat2.hu_moments.size()); ++i) {
        float diff = std::abs(static_cast<float>(feat1.hu_moments[i] - feat2.hu_moments[i]));
        similarity += 1.0f / (1.0f + diff);
    }
    
    return similarity / std::min(feat1.hu_moments.size(), feat2.hu_moments.size());
}

std::vector<float> GeometricFeatureCalculator::computeZernikeMoments(const cv::Mat& image, int max_order) {
    // Zernike矩的简化实现
    std::vector<float> moments;
    
    // 这里可以实现完整的Zernike矩计算
    // 为简化，返回空向量
    return moments;
}

float GeometricFeatureCalculator::assessGeometricQuality(const GeometricFeatures& features) {
    // 基于面积和紧致度评估几何特征质量
    float area_score = std::min(1.0f, features.area / 1000.0f);
    float compactness_score = features.compactness;
    
    return 0.6f * area_score + 0.4f * compactness_score;
}

} // namespace VideoMatcher