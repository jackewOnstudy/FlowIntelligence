#include "video_matcher.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>

namespace VideoMatcher {

void VideoMatcherUtils::saveNpyFiles(const std::string& video_path, const cv::Mat& motion_data, 
                                     const cv::Size& grid_size, const std::string& output_path) {
    // 对应Python的save_npy_files函数
    std::filesystem::path path(video_path);
    std::string file_name = path.stem().string();
    std::string file_folder = file_name.substr(0, file_name.length() - 1);
    std::string side = file_name.substr(file_name.length() - 1);
    
    std::string f_grid = std::to_string(grid_size.width) + "x" + std::to_string(grid_size.height);
    std::string output_folder = output_path + "/" + file_folder + "/" + side + "/";
    
    std::filesystem::create_directories(output_folder);
    std::string output_file = output_folder + f_grid + ".xml";
    
    cv::FileStorage fs(output_file, cv::FileStorage::WRITE);
    fs << "motion_data" << motion_data;
    fs.release();
    
    std::cout << output_file << " Saved!" << std::endl;
}

cv::Mat VideoMatcherUtils::loadNpyFiles(const std::string& video_path, const cv::Size& grid_size, 
                                        const std::string& output_path) {
    // 对应Python的load_npy_files函数
    std::filesystem::path path(video_path);
    std::string file_name = path.stem().string();
    std::string file_folder = file_name.substr(0, file_name.length() - 1);
    std::string side = file_name.substr(file_name.length() - 1);
    
    std::string f_grid = std::to_string(grid_size.width) + "x" + std::to_string(grid_size.height);
    std::string load_file = output_path + "/" + file_folder + "/" + side + "/" + f_grid + ".xml";
    
    cv::Mat motion_data;
    cv::FileStorage fs(load_file, cv::FileStorage::READ);
    fs["motion_data"] >> motion_data;
    fs.release();
    
    std::cout << load_file << " Loaded!" << std::endl;
    return motion_data;
}

cv::Mat VideoMatcherUtils::getMotionCountWithShiftingGrid(const std::string& video_path, 
                                                         const cv::Size& stride, const cv::Size& grid_size,
                                                         const Parameters& params, int& num_cols, int& num_rows) {
    // 对应Python的get_motion_count_with_shifting_grid_visualization函数
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Error: Could not open video: " + video_path);
    }
    
    cv::Size GaussianBlurKernel = params.GaussianBlurKernel;
    int Binary_threshold = params.Binary_threshold;
    
    int stride_w = stride.width, stride_h = stride.height;
    int grid_w = grid_size.width, grid_h = grid_size.height;
    
    // 获取视频帧的宽和高和帧数
    int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    // 计算网格数量
    num_cols = (frame_w - grid_w) / stride_w + 1;
    num_rows = (frame_h - grid_h) / stride_h + 1;
    int total_grids = num_cols * num_rows;
    
    std::filesystem::path path(video_path);
    std::string file_name = path.stem().string();
    
    std::cout << "Processing " << file_name << " frames..." << std::endl;
    
    // 设置初始帧
    cv::Mat frame, prev_frame_gray;
    cap >> frame;
    cv::cvtColor(frame, prev_frame_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(prev_frame_gray, prev_frame_gray, GaussianBlurKernel, 0);
    
    std::vector<std::vector<int>> motion_timestamps_per_grid;
    
    // // 根据网格大小动态调整处理帧数
    int effective_max_frames = params.max_frames;
    // if (grid_size.width <= 8 && grid_size.height <= 8) {
    //     effective_max_frames = std::min(500, params.max_frames);  // 8x8网格减少到500帧
    // }
    
    std::cout << "Processing up to " << effective_max_frames << " frames for grid size " 
              << grid_size.width << "x" << grid_size.height << std::endl;
    
    // 处理指定帧数或直到视频结束
    for (int frame_idx = 0; frame_idx < effective_max_frames && cap.read(frame); frame_idx += 2) {
        // 隔帧读取
        if (!cap.read(frame)) break;
        
        cv::Mat frame_gray;
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(frame_gray, frame_gray, GaussianBlurKernel, 0);
        
        // 计算当前帧和前一帧的差异
        cv::Mat frame_diff;
        cv::absdiff(prev_frame_gray, frame_gray, frame_diff);
        cv::GaussianBlur(frame_diff, frame_diff, GaussianBlurKernel, 0);
        
        // 帧间差分结果阈值
        cv::Mat binary_diff;
        cv::threshold(frame_diff, binary_diff, Binary_threshold, 1, cv::THRESH_BINARY);
        
        std::vector<int> motion_timestamps_grid;
        
        // 遍历每个小网格
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < num_cols; ++j) {
                // 计算当前网格的坐标
                int grid_x = j * stride_w;
                int grid_y = i * stride_h;
                
                // 提取当前网格区域
                cv::Rect grid_rect(grid_x, grid_y, grid_w, grid_h);
                cv::Mat grid_region = binary_diff(grid_rect);
                
                // 计算区域内运动像素数量
                int motion_sum = cv::sum(grid_region)[0];
                motion_timestamps_grid.push_back(motion_sum);
            }
        }
        
        motion_timestamps_per_grid.push_back(motion_timestamps_grid);
        prev_frame_gray = frame_gray.clone();
        
        if (frame_idx % 100 == 0) {
            std::cout << "Processed " << frame_idx << " frames" << std::endl;
        }
    }
    
    cap.release();
    
    // 转换为cv::Mat格式，每行对应一个网格的时间序列
    cv::Mat result(total_grids, motion_timestamps_per_grid.size(), CV_32S);
    
    for (int grid = 0; grid < total_grids; ++grid) {
        for (size_t frame = 0; frame < motion_timestamps_per_grid.size(); ++frame) {
            result.at<int>(grid, frame) = motion_timestamps_per_grid[frame][grid];
        }
    }
    
    std::cout << "Motion count extraction completed." << std::endl;
    return result;
}

cv::Mat VideoMatcherUtils::get4nGridMotionCount(const cv::Mat& motion_count_per_grid, 
                                               int& col_grid_num, int& row_grid_num, bool shifting_flag) {
    // 对应Python的get_4n_grid_motion_count函数
    int total_grids = col_grid_num * row_grid_num;
    int seq_len = motion_count_per_grid.cols;
    int actual_grid_count = motion_count_per_grid.rows;
    
    int new_col_grid_num = col_grid_num / 2;
    int new_row_grid_num = row_grid_num / 2;
    
    std::vector<cv::Mat> new_motion_status;
    
    for (int k = 0; k < new_col_grid_num * new_row_grid_num; ++k) {
        int row_idx, col_idx;
        
        // if (shifting_flag) {
        //     row_idx = k / new_col_grid_num;
        //     col_idx = k % new_col_grid_num;
        // } else {
        //     row_idx = (k / new_col_grid_num) * 2;
        //     col_idx = (k % new_col_grid_num) * 2;
        // }

        row_idx = (k / new_col_grid_num) * 2;
        col_idx = (k % new_col_grid_num) * 2;
        
        int m = row_idx * col_grid_num + col_idx;
        
        int m1, m2, m3, m4;
        if (shifting_flag) {
            m1 = m;
            m2 = m + 2;
            m3 = m + col_grid_num * 2;
            m4 = m + (col_grid_num + 1) * 2;
        } else {
            m1 = m;
            m2 = m + 1;
            m3 = m + col_grid_num;
            m4 = m + col_grid_num + 1;
        }
        
        // 确保索引在范围内
        if (m4 < actual_grid_count) {
            // 将4个小网格的运动元素个数相加
            cv::Mat combined_status = motion_count_per_grid.row(m1) + 
                                     motion_count_per_grid.row(m2) + 
                                     motion_count_per_grid.row(m3) + 
                                     motion_count_per_grid.row(m4);
            new_motion_status.push_back(combined_status);
        }
    }
    
    // 将结果组合成一个矩阵
    cv::Mat result(new_motion_status.size(), seq_len, motion_count_per_grid.type());
    for (size_t i = 0; i < new_motion_status.size(); ++i) {
        new_motion_status[i].copyTo(result.row(i));
    }
    
    col_grid_num = new_col_grid_num;
    row_grid_num = new_row_grid_num;
    
    return result;
}

cv::Mat VideoMatcherUtils::getMotionStatus(const cv::Mat& motion_count, int motion_threshold) {
    // 对应Python的get_motion_status函数
    cv::Mat motion_count_f;
    motion_count.convertTo(motion_count_f, CV_32F);  // 转换为浮点类型
    
    cv::Mat motion_status;
    cv::threshold(motion_count_f, motion_status, motion_threshold, 1, cv::THRESH_BINARY);
    motion_status.convertTo(motion_status, CV_8U);
    return motion_status;
}

cv::Mat VideoMatcherUtils::getMotionCountWithOtsu(const std::string& video_path, 
                                                  const cv::Size& stride, const cv::Size& grid_size,
                                                  const Parameters& params, int& num_cols, int& num_rows) {
    // 对应Python的get_motion_count_with_otsu函数
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Error: Could not open video: " + video_path);
    }
    
    cv::Size GaussianBlurKernel = params.GaussianBlurKernel;
    
    int stride_w = stride.width, stride_h = stride.height;
    int grid_w = grid_size.width, grid_h = grid_size.height;
    
    // 获取视频帧的宽和高和帧数
    int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    
    // 计算网格数量
    num_cols = (frame_w - grid_w) / stride_w + 1;
    num_rows = (frame_h - grid_h) / stride_h + 1;
    int total_grids = num_cols * num_rows;
    
    std::filesystem::path path(video_path);
    std::string file_name = path.stem().string();
    
    std::cout << "Processing " << file_name << " frames with Otsu thresholding..." << std::endl;
    
    int effective_max_frames = params.max_frames;
    
    // 设置初始帧
    cv::Mat frame, prev_frame_gray;
    cap >> frame;
    cv::cvtColor(frame, prev_frame_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(prev_frame_gray, prev_frame_gray, GaussianBlurKernel, 0);
    
    std::vector<std::vector<int>> motion_timestamps_per_grid;
    
    // 处理指定帧数或直到视频结束
    for (int frame_idx = 0; frame_idx < effective_max_frames && cap.read(frame); frame_idx += 2) {
        // 隔帧读取
        if (!cap.read(frame)) break;
        
        cv::Mat frame_gray;
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(frame_gray, frame_gray, GaussianBlurKernel, 0);
        
        // 计算当前帧和前一帧的差异
        cv::Mat frame_diff;
        cv::absdiff(prev_frame_gray, frame_gray, frame_diff);
        cv::GaussianBlur(frame_diff, frame_diff, GaussianBlurKernel, 0);
        
        // 使用Otsu自动计算阈值进行帧间差分二值化
        cv::Mat binary_diff;
        cv::threshold(frame_diff, binary_diff, 0, 1, cv::THRESH_BINARY + cv::THRESH_OTSU);
        
        std::vector<int> motion_timestamps_grid;
        
        // 遍历每个小网格
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < num_cols; ++j) {
                // 计算当前网格的坐标
                int grid_x = j * stride_w;
                int grid_y = i * stride_h;
                
                // 提取当前网格区域
                cv::Rect grid_rect(grid_x, grid_y, grid_w, grid_h);
                cv::Mat grid_region = binary_diff(grid_rect);
                
                // 计算区域内运动像素数量
                int motion_sum = cv::sum(grid_region)[0];
                motion_timestamps_grid.push_back(motion_sum);
            }
        }
        
        motion_timestamps_per_grid.push_back(motion_timestamps_grid);
        prev_frame_gray = frame_gray.clone();
        
        if (frame_idx % 100 == 0) {
            std::cout << "Processed " << frame_idx << " frames" << std::endl;
        }
    }
    
    cap.release();
    
    // 转换为cv::Mat格式，每行对应一个网格的时间序列
    cv::Mat result(total_grids, motion_timestamps_per_grid.size(), CV_32S);
    
    for (int grid = 0; grid < total_grids; ++grid) {
        for (size_t frame = 0; frame < motion_timestamps_per_grid.size(); ++frame) {
            result.at<int>(grid, frame) = motion_timestamps_per_grid[frame][grid];
        }
    }
    
    std::cout << "Motion count extraction with Otsu completed." << std::endl;
    return result;
}

double VideoMatcherUtils::calculateOtsuThreshold(const cv::Mat& data) {
    // 计算Otsu阈值的简化版本
    double minVal, maxVal;
    cv::minMaxLoc(data, &minVal, &maxVal);
    
    if (maxVal == minVal) {
        return minVal;
    }
    
    // 使用OpenCV内置的Otsu阈值计算
    cv::Mat data_8u;
    data.convertTo(data_8u, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    
    double threshold_8u = cv::threshold(data_8u, cv::Mat(), 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    
    // 将阈值转换回原始数据范围
    double threshold = minVal + (threshold_8u / 255.0) * (maxVal - minVal);
    
    return threshold;
}

cv::Mat VideoMatcherUtils::getMotionStatusWithOtsu(const cv::Mat& motion_count, float min_threshold) {
    // 对应Python的get_motion_status_with_otsu函数
    // 网格内自适应阈值
    int num_grids = motion_count.rows;
    int seq_len = motion_count.cols;
    
    cv::Mat motion_status = cv::Mat::zeros(num_grids, seq_len, CV_8U);
    
    for (int grid_idx = 0; grid_idx < num_grids; ++grid_idx) {
        cv::Mat counts = motion_count.row(grid_idx);
        
        // 检查数据是否有效（避免全零或单一值）
        double minVal, maxVal;
        cv::minMaxLoc(counts, &minVal, &maxVal);
        
        double threshold;
        if (maxVal == minVal) {
            // 如果所有值都相同，直接使用最小阈值
            threshold = min_threshold;
        } else {
            try {
                threshold = calculateOtsuThreshold(counts);
            } catch (...) {
                std::cout << "Warning: Otsu threshold failed for grid " << grid_idx << ", using min_threshold." << std::endl;
                threshold = min_threshold;
            }
        }
        
        threshold = std::max(threshold, static_cast<double>(min_threshold));
        
        // 二值化：大于阈值的判为 1（有运动），否则为 0
        for (int j = 0; j < seq_len; ++j) {
            motion_status.at<uchar>(grid_idx, j) = (counts.at<int>(0, j) > threshold) ? 1 : 0;
        }
    }
    
    return motion_status;
}

cv::Mat VideoMatcherUtils::getMotionStatusGlobalOtsu(const cv::Mat& motion_count, float min_threshold) {
    // 对应Python的get_motion_status_global_otsu函数
    // 全局自适应阈值
    int num_grids = motion_count.rows;
    int seq_len = motion_count.cols;
    
    cv::Mat motion_status = cv::Mat::zeros(num_grids, seq_len, CV_8U);
    
    // 对每一帧进行全局Otsu阈值计算
    for (int t = 0; t < seq_len; ++t) {
        cv::Mat counts = motion_count.col(t);
        
        // 检查数据是否有效
        double minVal, maxVal;
        cv::minMaxLoc(counts, &minVal, &maxVal);
        
        double threshold;
        if (maxVal == minVal) {
            threshold = min_threshold;
        } else {
            try {
                threshold = calculateOtsuThreshold(counts);
            } catch (...) {
                std::cout << "Warning: Global Otsu threshold failed for frame " << t << ", using min_threshold." << std::endl;
                threshold = min_threshold;
            }
        }
        
        threshold = std::max(threshold, static_cast<double>(min_threshold));
        
        // 根据阈值标记所有网格
        for (int grid_idx = 0; grid_idx < num_grids; ++grid_idx) {
            motion_status.at<uchar>(grid_idx, t) = (counts.at<int>(grid_idx, 0) > threshold) ? 1 : 0;
        }
    }
    
    return motion_status;
}

std::vector<MatchTriplet> VideoMatcherUtils::processTriplets(const std::vector<MatchTriplet>& triplets) {
    // 对应Python的process_triplets函数
    std::set<int> used_p2;
    std::vector<MatchTriplet> result;
    std::map<int, std::vector<MatchTriplet>> p1_groups;
    
    // 第一遍遍历，按p1值分组
    for (const auto& triplet : triplets) {
        p1_groups[triplet.grid1].push_back(triplet);
    }
    
    // 按照每组最小dist的顺序处理
    std::vector<std::pair<float, int>> group_min_dists;
    for (const auto& pair : p1_groups) {
        float min_dist = pair.second[0].distance;
        for (const auto& triplet : pair.second) {
            if (triplet.distance < min_dist) {
                min_dist = triplet.distance;
            }
        }
        group_min_dists.emplace_back(min_dist, pair.first);
    }
    
    std::sort(group_min_dists.begin(), group_min_dists.end());
    
    // 处理每个p1组
    for (const auto& pair : group_min_dists) {
        int p1 = pair.second;
        const auto& triplets_group = p1_groups[p1];
        
        // 过滤掉p2已使用的三元组
        std::vector<MatchTriplet> valid_triplets;
        for (const auto& triplet : triplets_group) {
            if (used_p2.find(triplet.grid2) == used_p2.end()) {
                valid_triplets.push_back(triplet);
            }
        }
        
        if (!valid_triplets.empty()) {
            // 找到距离最小的
            auto min_it = std::min_element(valid_triplets.begin(), valid_triplets.end(),
                [](const MatchTriplet& a, const MatchTriplet& b) {
                    return a.distance < b.distance;
                });
            result.push_back(*min_it);
            used_p2.insert(min_it->grid2);
        }
    }
    
    return result;
}

void VideoMatcherUtils::matchResultView(const std::string& path, const std::vector<MatchTriplet>& match_result,
                                       const cv::Size& grid_size, const cv::Size& stride, 
                                       const std::string& output_path, int which_video) {
    // 对应Python的match_result_view函数
    std::filesystem::path video_path(path);
    std::string folder = video_path.parent_path().string();
    std::string file = video_path.stem().string();
    
    std::string first_frame_dir = folder + "/first_frame";
    std::string frame_path = first_frame_dir + "/" + file + ".jpg";
    
    cv::Mat frame = cv::imread(frame_path);
    if (frame.empty()) {
        std::cerr << "Error: Could not open frame: " << frame_path << std::endl;
        return;
    }
    
    int frame_w = frame.cols;
    int frame_h = frame.rows;
    int stride_w = stride.width, stride_h = stride.height;
    int grid_w = grid_size.width, grid_h = grid_size.height;
    
    int num_cols = (frame_w - grid_w) / stride_w + 1;
    int num_rows = (frame_h - grid_h) / stride_h + 1;
    
    for (const auto& match : match_result) {
        int index1 = match.grid1;
        int index2 = match.grid2;
        float point = 0.5f;
        
        if (which_video == 1) {
            int i1 = index1 / num_cols;
            int j1 = index1 % num_cols;
            int grid_x = j1 * stride_w;
            int grid_y = i1 * stride_h;
            
            // 添加红色滤镜
            cv::Mat roi = frame(cv::Rect(grid_x, grid_y, grid_w, grid_h));
            cv::Mat red_mask = cv::Mat::ones(roi.size(), roi.type());
            red_mask.setTo(cv::Scalar(0, 0, 255));
            
            cv::addWeighted(roi, 1.0f - point, red_mask, point, 0, roi);
        }
        
        if (which_video == 2) {
            int i2 = index2 / num_cols;
            int j2 = index2 % num_cols;
            int grid_x = j2 * stride_w;
            int grid_y = i2 * stride_h;
            
            // 添加绿色滤镜
            cv::Mat roi = frame(cv::Rect(grid_x, grid_y, grid_w, grid_h));
            cv::Mat green_mask = cv::Mat::ones(roi.size(), roi.type());
            green_mask.setTo(cv::Scalar(0, 255, 0));
            
            cv::addWeighted(roi, 1.0f - point, green_mask, point, 0, roi);
        }
    }
    
    std::string output_folder = output_path + "/" + file;
    std::filesystem::create_directories(output_folder);
    std::string output_file = output_folder + "/" + std::to_string(grid_size.width) + "x" + 
                             std::to_string(grid_size.height) + "_match_result.jpg";
    
    cv::imwrite(output_file, frame);
    std::cout << "Save the match result to " << output_file << std::endl;
}

std::set<int> VideoMatcherUtils::getSmallIndexInLarge(int K, int num_large_grids_per_row, int num_small_grids_per_row,
                                                     int n_width, int n_height, bool shifting_flag) {
    // 对应Python的get_small_index_in_large函数
    // 计算大网格的行和列
    int large_grid_row = K / num_large_grids_per_row;
    int large_grid_col = K % num_large_grids_per_row;
    
    int small_grid_row = large_grid_row * n_height;
    int small_grid_col = large_grid_col * n_width;
    
    if (shifting_flag) {
        small_grid_row = small_grid_row - 1;
        small_grid_col = small_grid_col - 1;
        n_height += 1;
        n_width += 1;
    }
    
    std::set<int> small_grid_set;
    
    // 计算小网格的最终索引
    for (int i = 0; i < n_height; ++i) {
        for (int j = 0; j < n_width; ++j) {
            int small_grid_index = (small_grid_row + i) * num_small_grids_per_row + (small_grid_col + j);
            small_grid_set.insert(small_grid_index);
        }
    }
    
    return small_grid_set;
}

std::map<int, std::set<int>> VideoMatcherUtils::getSmallGridIndex(const std::vector<MatchTriplet>& match_result,
                                                                 const cv::Size& image_size, const cv::Size& small_grid_size,
                                                                 const cv::Size& large_grid_size, bool shifting_flag, int which_video) {
    // 对应Python的get_small_grid_index函数
    // 计算小网格和大网格的数量
    int num_small_grids_per_row = image_size.width / small_grid_size.width;
    int num_large_grids_per_row = image_size.width / large_grid_size.width;
    
    if (shifting_flag) {
        num_small_grids_per_row = num_small_grids_per_row * 2 - 1;
        num_large_grids_per_row = num_large_grids_per_row * 2 - 1;
    }
    
    // 计算对应的小网格的行和列
    int n_width = large_grid_size.width / small_grid_size.width;
    int n_height = large_grid_size.height / small_grid_size.height;
    
    if (which_video == 2) {
        std::map<int, std::set<int>> large_grid_corre_small_dict;
        
        for (const auto& triplet : match_result) {
            int index1 = triplet.grid1;
            int index2 = triplet.grid2;
            
            std::set<int> small_grid_index_set = getSmallIndexInLarge(
                index2, num_large_grids_per_row, num_small_grids_per_row, n_width, n_height, shifting_flag);
            
            if (large_grid_corre_small_dict.find(index1) != large_grid_corre_small_dict.end()) {
                // 合并集合
                large_grid_corre_small_dict[index1].insert(small_grid_index_set.begin(), small_grid_index_set.end());
            } else {
                large_grid_corre_small_dict[index1] = small_grid_index_set;
            }
        }
        
        return large_grid_corre_small_dict;
    }
    
    return std::map<int, std::set<int>>();
}

} // namespace VideoMatcher 