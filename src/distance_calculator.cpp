#include "video_matcher.h"
#include <algorithm>
#include <numeric>
#include <immintrin.h>  

namespace VideoMatcher {

class SIMDDistanceCalculator {
public:
    static int logicalAndCount_AVX2(const std::vector<bool>& x, const std::vector<bool>& y) {
        if (x.size() != y.size()) return 0;
        
        size_t length = x.size();
        int count = 0;

        std::vector<uint8_t> x_bytes(length), y_bytes(length);
        for (size_t i = 0; i < length; ++i) {
            x_bytes[i] = x[i] ? 1 : 0;
            y_bytes[i] = y[i] ? 1 : 0;
        }
        
        size_t simd_length = length & ~31; 
        
        for (size_t i = 0; i < simd_length; i += 32) {
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&x_bytes[i]));
            __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&y_bytes[i]));
            __m256i and_result = _mm256_and_si256(v1, v2);
            
            count += _mm_popcnt_u64(_mm256_extract_epi64(and_result, 0));
            count += _mm_popcnt_u64(_mm256_extract_epi64(and_result, 1));
            count += _mm_popcnt_u64(_mm256_extract_epi64(and_result, 2));
            count += _mm_popcnt_u64(_mm256_extract_epi64(and_result, 3));
        }
        
        for (size_t i = simd_length; i < length; ++i) {
            if (x[i] && y[i]) count++;
        }
        
        return count;
    }
    
    static std::pair<int, int> countTrue_AVX2(const std::vector<bool>& x, const std::vector<bool>& y) {
        if (x.size() != y.size()) return {0, 0};
        
        size_t length = x.size();
        int count_x = 0, count_y = 0;
        
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
    if (x.size() != y.size()) {
        throw std::invalid_argument("两个序列长度应相同");
    }
    
    if (x.empty()) {
        return 0.0f;
    }
    
    int and_count = SIMDDistanceCalculator::logicalAndCount_AVX2(x, y);
    auto [sum_x, sum_y] = SIMDDistanceCalculator::countTrue_AVX2(x, y);
            
    if (sum_x == 0 && sum_y == 0) {
        return 0.0f;
    }
    
    return 1.0f - (2.0f * and_count) / (sum_x + sum_y);
}
        
float DistanceCalculator::logicXorDistance(const std::vector<bool>& x, const std::vector<bool>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("两个序列长度应相同");
    }
    
    if (x.empty()) {
        return 0.0f;
    }
    
    size_t length = x.size();
    int xor_count = 0;
    
    std::vector<uint8_t> x_bytes(length), y_bytes(length);
    for (size_t i = 0; i < length; ++i) {
        x_bytes[i] = x[i] ? 1 : 0;
        y_bytes[i] = y[i] ? 1 : 0;
    }
    
    size_t simd_length = length & ~31;
    
    for (size_t i = 0; i < simd_length; i += 32) {
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&x_bytes[i]));
        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&y_bytes[i]));
        __m256i xor_result = _mm256_xor_si256(v1, v2);
        
        xor_count += _mm_popcnt_u64(_mm256_extract_epi64(xor_result, 0));
        xor_count += _mm_popcnt_u64(_mm256_extract_epi64(xor_result, 1));
        xor_count += _mm_popcnt_u64(_mm256_extract_epi64(xor_result, 2));
        xor_count += _mm_popcnt_u64(_mm256_extract_epi64(xor_result, 3));
    }
    
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
    if (distance_metric == "logic_and") {
        return logicAndDistance(segment1, segment2);
    } else if (distance_metric == "logic_xonr") {
        return logicXorDistance(segment1, segment2);
    } else if (distance_metric == "hamming") {
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