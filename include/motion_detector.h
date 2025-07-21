#pragma once

#include "video_matcher.h"
#include <opencv2/opencv.hpp>

namespace VideoMatcher {

class MotionDetector {
private:
    cv::Ptr<cv::BackgroundSubtractorMOG2> bg_subtractor_;
    int motion_threshold_;
    
public:
    explicit MotionDetector(int motion_threshold = 30);
    
    // 帧间差分法检测运动
    cv::Mat detectMotion(const cv::Mat& frame1, const cv::Mat& frame2);
    
    // 构建patch运动状态序列
    std::vector<PatchInfo> buildMotionSequence(
        const std::vector<cv::Mat>& frames,
        int patch_size, 
        int step_size, 
        int threshold
    );
    
    // 计算单个patch的运动像素数
    int countMotionPixels(const cv::Mat& motion_mask, const cv::Rect& patch_area);
    
    // 设置运动检测阈值
    void setMotionThreshold(int threshold) { motion_threshold_ = threshold; }
};

} // namespace VideoMatcher 