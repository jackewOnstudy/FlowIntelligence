#pragma once

#include "video_matcher.h"
#include "distance_calculator.h"
#include <random>

namespace VideoMatcher {

class MatchPropagator {
private:
    std::shared_ptr<DistanceCalculator> distance_calculator_;
    float relaxed_threshold_;
    bool enable_random_search_;
    std::mt19937 random_engine_;
    
public:
    MatchPropagator(
        std::shared_ptr<DistanceCalculator> distance_calc,
        float relaxed_threshold,
        bool enable_random_search = false
    );
    
    // 传播匹配结果
    std::vector<MatchResult> propagateMatches(
        const std::vector<MatchResult>& initial_matches,
        const std::vector<PatchInfo>& patches1,
        const std::vector<PatchInfo>& patches2,
        int step_size
    );
    
private:
    // 搜索邻域匹配
    std::vector<MatchResult> searchNeighborhood(
        const MatchResult& match,
        const std::vector<PatchInfo>& patches1,
        const std::vector<PatchInfo>& patches2,
        int search_radius
    );
    
    // 随机搜索
    std::vector<MatchResult> randomSearch(
        const MatchResult& match,
        const std::vector<PatchInfo>& patches1,
        const std::vector<PatchInfo>& patches2
    );
    
    // 计算指数递减概率
    float exponentialDecayProbability(float distance, float base_prob = 0.1f);
    
    // 获取邻域patches
    std::vector<const PatchInfo*> getNeighborPatches(
        const PatchInfo& center_patch,
        const std::vector<PatchInfo>& all_patches,
        int radius
    );
};

} // namespace VideoMatcher 