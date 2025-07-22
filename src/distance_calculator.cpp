#include "video_matcher.h"
#include <algorithm>
#include <numeric>
#include <immintrin.h>  // 新增：SIMD支持

namespace VideoMatcher {

// 新增：SIMD优化的逻辑与计算
class SIMDDistanceCalculator {
public:
    // AVX2优化的逻辑与计算
    static int logicalAndCount_AVX2(const std::vector<bool>& x, const std::vector<bool>& y) {
        if (x.size() != y.size()) return 0;
        
        size_t length = x.size();
        int count = 0;
        
        // 将std::vector<bool>转换为连续的字节数组以支持SIMD
        std::vector<uint8_t> x_bytes(length), y_bytes(length);
        for (size_t i = 0; i < length; ++i) {
            x_bytes[i] = x[i] ? 1 : 0;
            y_bytes[i] = y[i] ? 1 : 0;
        }
        
        size_t simd_length = length & ~31;  // 32的倍数
        
        // SIMD处理
        for (size_t i = 0; i < simd_length; i += 32) {
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&x_bytes[i]));
            __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&y_bytes[i]));
            __m256i and_result = _mm256_and_si256(v1, v2);
            
            // 计算设置位的数量
            count += _mm_popcnt_u64(_mm256_extract_epi64(and_result, 0));
            count += _mm_popcnt_u64(_mm256_extract_epi64(and_result, 1));
            count += _mm_popcnt_u64(_mm256_extract_epi64(and_result, 2));
            count += _mm_popcnt_u64(_mm256_extract_epi64(and_result, 3));
        }
        
        // 处理剩余元素
        for (size_t i = simd_length; i < length; ++i) {
            if (x[i] && y[i]) count++;
        }
        
        return count;
    }
    
    // AVX2优化的计数
    static std::pair<int, int> countTrue_AVX2(const std::vector<bool>& x, const std::vector<bool>& y) {
        if (x.size() != y.size()) return {0, 0};
        
        size_t length = x.size();
        int count_x = 0, count_y = 0;
        
        // 转换为字节数组
        std::vector<uint8_t> x_bytes(length), y_bytes(length);
        for (size_t i = 0; i < length; ++i) {
            x_bytes[i] = x[i] ? 1 : 0;
            y_bytes[i] = y[i] ? 1 : 0;
        }
        
        size_t simd_length = length & ~31;
        
        for (size_t i = 0; i < simd_length; i += 32) {
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&x_bytes[i]));
            __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&y_bytes[i]));
            
            count_x += _mm_popcnt_u64(_mm256_extract_epi64(v1, 0));
            count_x += _mm_popcnt_u64(_mm256_extract_epi64(v1, 1));
            count_x += _mm_popcnt_u64(_mm256_extract_epi64(v1, 2));
            count_x += _mm_popcnt_u64(_mm256_extract_epi64(v1, 3));
            
            count_y += _mm_popcnt_u64(_mm256_extract_epi64(v2, 0));
            count_y += _mm_popcnt_u64(_mm256_extract_epi64(v2, 1));
            count_y += _mm_popcnt_u64(_mm256_extract_epi64(v2, 2));
            count_y += _mm_popcnt_u64(_mm256_extract_epi64(v2, 3));
        }
        
        for (size_t i = simd_length; i < length; ++i) {
            if (x[i]) count_x++;
            if (y[i]) count_y++;
        }
        
        return {count_x, count_y};
    }
};

float DistanceCalculator::logicAndDistance(const std::vector<bool>& x, const std::vector<bool>& y) {
    // 对应Python的logic_and_distance函数
    if (x.size() != y.size()) {
        throw std::invalid_argument("两个序列长度应相同");
    }
    
    if (x.empty()) {
        return 0.0f;
    }
    
    // 使用SIMD优化的计算
    int and_count = SIMDDistanceCalculator::logicalAndCount_AVX2(x, y);
    auto [sum_x, sum_y] = SIMDDistanceCalculator::countTrue_AVX2(x, y);
            
    if (sum_x == 0 && sum_y == 0) {
        return 0.0f;
    }
    
    // Python版本的计算公式：1 - np.sum(np.logical_and(x, y)) / (np.sum(x) + np.sum(y)) * 2
    return 1.0f - (2.0f * and_count) / (sum_x + sum_y);
}
        
float DistanceCalculator::logicXorDistance(const std::vector<bool>& x, const std::vector<bool>& y) {
    // 对应Python的logic_xonr_distance函数 - 使用SIMD优化
    if (x.size() != y.size()) {
        throw std::invalid_argument("两个序列长度应相同");
    }
    
    if (x.empty()) {
        return 0.0f;
    }
    
    size_t length = x.size();
    int xor_count = 0;
    
    // 转换为字节数组支持SIMD
    std::vector<uint8_t> x_bytes(length), y_bytes(length);
    for (size_t i = 0; i < length; ++i) {
        x_bytes[i] = x[i] ? 1 : 0;
        y_bytes[i] = y[i] ? 1 : 0;
    }
    
    size_t simd_length = length & ~31;
    
    // SIMD XOR计算
    for (size_t i = 0; i < simd_length; i += 32) {
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&x_bytes[i]));
        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&y_bytes[i]));
        __m256i xor_result = _mm256_xor_si256(v1, v2);
        
        xor_count += _mm_popcnt_u64(_mm256_extract_epi64(xor_result, 0));
        xor_count += _mm_popcnt_u64(_mm256_extract_epi64(xor_result, 1));
        xor_count += _mm_popcnt_u64(_mm256_extract_epi64(xor_result, 2));
        xor_count += _mm_popcnt_u64(_mm256_extract_epi64(xor_result, 3));
    }
    
    // 处理剩余元素
    for (size_t i = simd_length; i < length; ++i) {
        if (x[i] != y[i]) {
            xor_count++;
        }
    }
    
    return static_cast<float>(xor_count);
}

float DistanceCalculator::segmentSimilarity(const std::vector<bool>& segment1, 
                                          const std::vector<bool>& segment2, 
                                          const std::string& distance_metric) {
    // 对应Python的segment_similarity函数
    if (distance_metric == "logic_and") {
        return logicAndDistance(segment1, segment2);
    } else if (distance_metric == "logic_xonr") {
        return logicXorDistance(segment1, segment2);
    } else if (distance_metric == "hamming") {
        // 使用SIMD优化的汉明距离
        if (segment1.size() != segment2.size()) {
            return 1.0f;
        }
        
        int diff_count = static_cast<int>(logicXorDistance(segment1, segment2));
        return static_cast<float>(diff_count) / segment1.size();
    } else {
        throw std::invalid_argument("Unsupported distance metric: " + distance_metric);
    }
}

} // namespace VideoMatcher