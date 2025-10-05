#include "video_matcher.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <unordered_set> 
#include <unordered_map>  
#ifdef _OPENMP
#include <omp.h>  
#endif

namespace VideoMatcher {

void VideoMatcherUtils::saveNpyFiles(const std::string& video_path, const cv::Mat& motion_data, 
                                     const cv::Size& grid_size, const std::string& output_path) {
    std::filesystem::path path(video_path);
    std::string file_name = path.stem().string();
    std::string output_folder = output_path + "/" + file_name + "/";
    std::string f_grid = std::to_string(grid_size.width) + "x" + std::to_string(grid_size.height);

    std::filesystem::create_directories(output_folder);
    std::string output_file = output_folder + f_grid + ".xml";
    
    cv::FileStorage fs(output_file, cv::FileStorage::WRITE);
    fs << "motion_data" << motion_data;
    fs.release();
    
    std::cout << output_file << " Saved!" << std::endl;
}

cv::Mat VideoMatcherUtils::loadNpyFiles(const std::string& video_path, const cv::Size& grid_size, 
                                        const std::string& output_path) {
    std::filesystem::path path(video_path);
    std::string file_name = path.stem().string();
    std::string f_grid = std::to_string(grid_size.width) + "x" + std::to_string(grid_size.height);
    std::string load_file = output_path + "/" + file_name + "/" + f_grid + ".xml";
    
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
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Error: Could not open video: " + video_path);
    }
    
    cv::Size GaussianBlurKernel = params.GaussianBlurKernel;
    int Binary_threshold = params.Binary_threshold;
    
    int stride_w = stride.width, stride_h = stride.height;
    int grid_w = grid_size.width, grid_h = grid_size.height;
    
    int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    (void)cap.get(cv::CAP_PROP_FRAME_COUNT); 

    num_cols = (frame_w - grid_w) / stride_w + 1;
    num_rows = (frame_h - grid_h) / stride_h + 1;
    int total_grids = num_cols * num_rows;
    
    std::filesystem::path path(video_path);
    std::string file_name = path.stem().string();
    
    std::cout << "Processing " << file_name << " frames..." << std::endl;
    
    std::vector<std::vector<int>> motion_timestamps_per_grid;
    motion_timestamps_per_grid.reserve(params.max_frames / 2); // 预分配内存
    
    cv::Mat frame, prev_frame_gray, frame_gray, frame_diff, binary_diff;
    cap >> frame;
    cv::cvtColor(frame, prev_frame_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(prev_frame_gray, prev_frame_gray, GaussianBlurKernel, 0);
    
    int effective_max_frames = params.max_frames;
    
    std::cout << "Processing up to " << effective_max_frames << " frames for grid size " 
              << grid_size.width << "x" << grid_size.height << std::endl;
    
    for (int frame_idx = 0; frame_idx < effective_max_frames && cap.read(frame); frame_idx += 2) {
        if (!cap.read(frame)) break;
        
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(frame_gray, frame_gray, GaussianBlurKernel, 0);
        
        cv::absdiff(prev_frame_gray, frame_gray, frame_diff);
        cv::GaussianBlur(frame_diff, frame_diff, GaussianBlurKernel, 0);
        
        cv::threshold(frame_diff, binary_diff, Binary_threshold, 1, cv::THRESH_BINARY);
        
        std::vector<int> motion_timestamps_grid(total_grids);
        
        #pragma omp parallel for collapse(2) schedule(dynamic) if(total_grids > 100)
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < num_cols; ++j) {
                int grid_x = j * stride_w;
                int grid_y = i * stride_h;
                
                cv::Rect grid_rect(grid_x, grid_y, grid_w, grid_h);
                cv::Mat grid_region = binary_diff(grid_rect);
                
                int motion_sum = cv::sum(grid_region)[0];
                int grid_index = i * num_cols + j;
                motion_timestamps_grid[grid_index] = motion_sum;
            }
        }
        
        motion_timestamps_per_grid.push_back(std::move(motion_timestamps_grid));
        prev_frame_gray = frame_gray.clone();
        
        if (frame_idx % 100 == 0) {
            std::cout << "Processed " << frame_idx << " frames" << std::endl;
        }
    }
    
    cap.release();

    cv::Mat result(total_grids, motion_timestamps_per_grid.size(), CV_32S);
    
    #pragma omp parallel for schedule(static) if(total_grids > 1000)
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
    (void)(col_grid_num * row_grid_num);
    int seq_len = motion_count_per_grid.cols;
    int actual_grid_count = motion_count_per_grid.rows;
    
    int new_col_grid_num = col_grid_num / 2;
    int new_row_grid_num = row_grid_num / 2;
    int new_total_grids = new_col_grid_num * new_row_grid_num;
    
    cv::Mat result(new_total_grids, seq_len, motion_count_per_grid.type());

    #pragma omp parallel for schedule(dynamic) if(new_total_grids > 100)
    for (int k = 0; k < new_total_grids; ++k) {
        int row_idx = (k / new_col_grid_num) * 2;
        int col_idx = (k % new_col_grid_num) * 2;
        
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
        
        if (m4 < actual_grid_count) {
            for (int col = 0; col < seq_len; ++col) {
                result.at<int>(k, col) = motion_count_per_grid.at<int>(m1, col) + 
                                       motion_count_per_grid.at<int>(m2, col) + 
                                       motion_count_per_grid.at<int>(m3, col) + 
                                       motion_count_per_grid.at<int>(m4, col);
            }
        }
    }
    
    col_grid_num = new_col_grid_num;
    row_grid_num = new_row_grid_num;
    
    return result;
}

cv::Mat VideoMatcherUtils::getMotionStatus(const cv::Mat& motion_count, int motion_threshold) {
    cv::Mat motion_status;
    
    // 将CV_32S类型转换为CV_32F类型，因为cv::threshold不支持CV_32S
    cv::Mat motion_count_float;
    motion_count.convertTo(motion_count_float, CV_32F);
    
    cv::threshold(motion_count_float, motion_status, static_cast<float>(motion_threshold), 1, cv::THRESH_BINARY);
    motion_status.convertTo(motion_status, CV_8U);
    return motion_status;
}

cv::Mat VideoMatcherUtils::getMotionCountWithOtsu(const std::string& video_path, 
                                                  const cv::Size& stride, const cv::Size& grid_size,
                                                  const Parameters& params, int& num_cols, int& num_rows) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        throw std::runtime_error("Error: Could not open video: " + video_path);
    }
    
    cv::Size GaussianBlurKernel = params.GaussianBlurKernel;
    
    int stride_w = stride.width, stride_h = stride.height;
    int grid_w = grid_size.width, grid_h = grid_size.height;
    
    int frame_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    (void)cap.get(cv::CAP_PROP_FRAME_COUNT); 
    
    num_cols = (frame_w - grid_w) / stride_w + 1;
    num_rows = (frame_h - grid_h) / stride_h + 1;
    int total_grids = num_cols * num_rows;
    
    std::filesystem::path path(video_path);
    std::string file_name = path.stem().string();
    
    std::cout << "Processing " << file_name << " frames with Otsu thresholding..." << std::endl;
    
    int effective_max_frames = params.max_frames;
    
    std::vector<std::vector<int>> motion_timestamps_per_grid;
    motion_timestamps_per_grid.reserve(effective_max_frames / 2);

    cv::Mat frame, prev_frame_gray, frame_gray, frame_diff, binary_diff;
    cap >> frame;
    cv::cvtColor(frame, prev_frame_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(prev_frame_gray, prev_frame_gray, GaussianBlurKernel, 0);

    for (int frame_idx = 0; frame_idx < effective_max_frames && cap.read(frame); frame_idx += 2) {
        if (!cap.read(frame)) break;
        
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(frame_gray, frame_gray, GaussianBlurKernel, 0);
        
        cv::absdiff(prev_frame_gray, frame_gray, frame_diff);
        cv::GaussianBlur(frame_diff, frame_diff, GaussianBlurKernel, 0);
        
        cv::threshold(frame_diff, binary_diff, 0, 1, cv::THRESH_BINARY + cv::THRESH_OTSU);
        
        std::vector<int> motion_timestamps_grid(total_grids);
        
        #pragma omp parallel for collapse(2) schedule(dynamic) if(total_grids > 100)
        for (int i = 0; i < num_rows; ++i) {
            for (int j = 0; j < num_cols; ++j) {
                int grid_x = j * stride_w;
                int grid_y = i * stride_h;
                
                cv::Rect grid_rect(grid_x, grid_y, grid_w, grid_h);
                cv::Mat grid_region = binary_diff(grid_rect);

                int motion_sum = cv::sum(grid_region)[0];
                int grid_index = i * num_cols + j;
                motion_timestamps_grid[grid_index] = motion_sum;
            }
        }
        
        motion_timestamps_per_grid.push_back(std::move(motion_timestamps_grid));
        prev_frame_gray = frame_gray.clone();
        
        if (frame_idx % 100 == 0) {
            std::cout << "Processed " << frame_idx << " frames" << std::endl;
        }
    }
    
    cap.release();

    cv::Mat result(total_grids, motion_timestamps_per_grid.size(), CV_32S);
    
    #pragma omp parallel for schedule(static) if(total_grids > 1000)
    for (int grid = 0; grid < total_grids; ++grid) {
        for (size_t frame = 0; frame < motion_timestamps_per_grid.size(); ++frame) {
            result.at<int>(grid, frame) = motion_timestamps_per_grid[frame][grid];
        }
    }
    
    std::cout << "Motion count extraction with Otsu completed." << std::endl;
    return result;
}

double VideoMatcherUtils::calculateOtsuThreshold(const cv::Mat& data) {
    double minVal, maxVal;
    cv::minMaxLoc(data, &minVal, &maxVal);
    
    if (maxVal == minVal) {
        return minVal;
    }

    if (data.type() == CV_8U) {
        return cv::threshold(data, cv::Mat(), 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    }
    
    cv::Mat data_8u;
    data.convertTo(data_8u, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
    
    double threshold_8u = cv::threshold(data_8u, cv::Mat(), 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

    double threshold = minVal + (threshold_8u / 255.0) * (maxVal - minVal);
    
    return threshold;
}

cv::Mat VideoMatcherUtils::getMotionStatusWithOtsu(const cv::Mat& motion_count, float min_threshold) {
    int num_grids = motion_count.rows;
    int seq_len = motion_count.cols;
    
    cv::Mat motion_status = cv::Mat::zeros(num_grids, seq_len, CV_8U);
    
    #pragma omp parallel for schedule(dynamic) if(num_grids > 50)
    for (int grid_idx = 0; grid_idx < num_grids; ++grid_idx) {
        cv::Mat counts = motion_count.row(grid_idx);
        
        double minVal, maxVal;
        cv::minMaxLoc(counts, &minVal, &maxVal);
        
        double threshold;
        if (maxVal == minVal) {
            threshold = min_threshold;
        } else {
            try {
                threshold = calculateOtsuThreshold(counts);
            } catch (...) {
                threshold = min_threshold;
            }
        }
        
        threshold = std::max(threshold, static_cast<double>(min_threshold));

        for (int j = 0; j < seq_len; ++j) {
            motion_status.at<uchar>(grid_idx, j) = (counts.at<int>(0, j) > threshold) ? 1 : 0;
        }
    }
    
    return motion_status;
}

cv::Mat VideoMatcherUtils::getMotionStatusGlobalOtsu(const cv::Mat& motion_count, float min_threshold) {
    int num_grids = motion_count.rows;
    int seq_len = motion_count.cols;
    
    cv::Mat motion_status = cv::Mat::zeros(num_grids, seq_len, CV_8U);
    
    #pragma omp parallel for schedule(dynamic) if(seq_len > 50)
    for (int t = 0; t < seq_len; ++t) {
        cv::Mat counts = motion_count.col(t);
        
        double minVal, maxVal;
        cv::minMaxLoc(counts, &minVal, &maxVal);
        
        double threshold;
        if (maxVal == minVal) {
            threshold = min_threshold;
        } else {
            try {
                threshold = calculateOtsuThreshold(counts);
            } catch (...) {
                threshold = min_threshold;
            }
        }
        
        threshold = std::max(threshold, static_cast<double>(min_threshold));
        
        for (int grid_idx = 0; grid_idx < num_grids; ++grid_idx) {
            motion_status.at<uchar>(grid_idx, t) = (counts.at<int>(grid_idx, 0) > threshold) ? 1 : 0;
        }
    }
    
    return motion_status;
}

std::vector<MatchTriplet> VideoMatcherUtils::processTriplets(const std::vector<MatchTriplet>& triplets) {
    if (triplets.empty()) return {};
    
    std::unordered_set<int> used_p2;  
    std::vector<MatchTriplet> result;
    result.reserve(triplets.size() / 2);  
    
    std::unordered_map<int, std::vector<MatchTriplet>> p1_groups;
    
    for (const auto& triplet : triplets) {
        p1_groups[triplet.grid1].push_back(triplet);
    }
    
    std::vector<std::pair<float, int>> group_min_dists;
    group_min_dists.reserve(p1_groups.size());
    
    for (const auto& pair : p1_groups) {
        float min_dist = std::min_element(pair.second.begin(), pair.second.end(),
            [](const MatchTriplet& a, const MatchTriplet& b) {
                return a.distance < b.distance;
            })->distance;
        group_min_dists.emplace_back(min_dist, pair.first);
    }
    
    std::sort(group_min_dists.begin(), group_min_dists.end());
    
    for (const auto& pair : group_min_dists) {
        int p1 = pair.second;
        const auto& triplets_group = p1_groups[p1];
        
        std::vector<MatchTriplet> valid_triplets;
        valid_triplets.reserve(triplets_group.size());
        
        for (const auto& triplet : triplets_group) {
            if (used_p2.find(triplet.grid2) == used_p2.end()) {
                valid_triplets.push_back(triplet);
            }
        }
        
        if (!valid_triplets.empty()) {
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

void VideoMatcherUtils::saveMatchResultList(const std::string& video_path, const std::vector<MatchTriplet>& match_result,
                                           const cv::Size& grid_size, const std::string& output_path) {
    std::filesystem::path video_path_obj(video_path);
    std::string file_name = video_path_obj.stem().string();
    
    std::string output_folder = output_path + "/" + file_name;
    std::filesystem::create_directories(output_folder);
    
    std::string f_grid = std::to_string(grid_size.width) + "x" + std::to_string(grid_size.height);
    std::string output_file = output_folder + "/" + f_grid + ".txt";
    
    std::ofstream file(output_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << output_file << std::endl;
        return;
    }
    
    file << match_result.size() << std::endl;
    
    for (const auto& triplet : match_result) {
        file << triplet.grid1 << " " << triplet.grid2 << " " << triplet.distance << std::endl;
    }
    
    file.close();
    std::cout << output_file << " Saved!" << std::endl;
}

std::vector<MatchTriplet> VideoMatcherUtils::loadMatchResultList(const std::string& video_path, const cv::Size& grid_size,
                                                                const std::string& output_path) {
    std::filesystem::path video_path_obj(video_path);
    std::string file_name = video_path_obj.stem().string();
    std::string file_folder = file_name.substr(0, file_name.length() - 1);
    
    std::string f_grid = std::to_string(grid_size.width) + "x" + std::to_string(grid_size.height);
    std::string load_file = output_path + "/" + file_folder + "/" + f_grid + ".txt";
    
    std::vector<MatchTriplet> match_result;
    std::ifstream file(load_file);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for reading: " << load_file << std::endl;
        return match_result;
    }
    
    size_t num_results;
    file >> num_results;
    match_result.reserve(num_results);
    
    for (size_t i = 0; i < num_results; ++i) {
        int grid1, grid2;
        float distance;
        if (file >> grid1 >> grid2 >> distance) {
            match_result.emplace_back(grid1, grid2, distance);
        }
    }
    
    file.close();
    std::cout << load_file << " Loaded!" << std::endl;
    return match_result;
}

void VideoMatcherUtils::matchResultView(const std::string& path, const std::vector<MatchTriplet>& match_result,
                                       const cv::Size& grid_size, const cv::Size& stride, 
                                       const std::string& output_path, int which_video) {
    if (match_result.empty()) return;  
    
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
    
    float point = 0.5f;
    cv::Scalar color = (which_video == 1) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    
    for (const auto& match : match_result) {
        int index = (which_video == 1) ? match.grid1 : match.grid2;
        
        int i = index / num_cols;
        int j = index % num_cols;
        int grid_x = j * stride_w;
        int grid_y = i * stride_h;
        
        if (grid_x + grid_w <= frame_w && grid_y + grid_h <= frame_h) {
            cv::Mat roi = frame(cv::Rect(grid_x, grid_y, grid_w, grid_h));
            cv::Mat color_mask = cv::Mat::ones(roi.size(), roi.type());
            color_mask.setTo(color);
            
            cv::addWeighted(roi, 1.0f - point, color_mask, point, 0, roi);
        }
    }
    
    std::string output_folder = output_path + "/" + file;
    std::filesystem::create_directories(output_folder);
    std::string output_file = output_folder + "/" + std::to_string(grid_size.width) + "x" + 
                             std::to_string(grid_size.height) + "_match_result.jpg";
    
    cv::imwrite(output_file, frame);
    std::cout << "Save the match result to " << output_file << std::endl;
}

void VideoMatcherUtils::combinedMatchResultView(const std::string& video_path1, const std::string& video_path2,
                                               const std::vector<MatchTriplet>& match_result,
                                               const cv::Size& grid_size1, const cv::Size& grid_size2,
                                               const cv::Size& stride1, const cv::Size& stride2,
                                               const std::string& output_path) {
    if (match_result.empty()) return;
    std::filesystem::path video_path1_obj(video_path1);
    std::filesystem::path video_path2_obj(video_path2);
    std::string file1 = video_path1_obj.stem().string();
    std::string file2 = video_path2_obj.stem().string();
    
    cv::VideoCapture cap1(video_path1);
    cv::VideoCapture cap2(video_path2);
    
    if (!cap1.isOpened() || !cap2.isOpened()) {
        std::cerr << "Error: Could not open videos: " << video_path1 << " or " << video_path2 << std::endl;
        return;
    }
    
    cv::Mat frame1, frame2;
    cap1 >> frame1; 
    cap1 >> frame1; 
    cap2 >> frame2;  
    cap2 >> frame2;  
    
    if (frame1.empty() || frame2.empty()) {
        std::cerr << "Error: Could not read second frame from videos" << std::endl;
        cap1.release();
        cap2.release();
        return;
    }
    
    cap1.release();
    cap2.release();
    
    cv::Mat result_frame1 = frame1.clone();
    cv::Mat result_frame2 = frame2.clone();
    int frame_w1 = result_frame1.cols;
    int frame_h1 = result_frame1.rows;
    int stride_w1 = stride1.width, stride_h1 = stride1.height;
    int grid_w1 = grid_size1.width, grid_h1 = grid_size1.height;
    int num_cols1 = (frame_w1 - grid_w1) / stride_w1 + 1;
    
    float point = 0.5f;
    cv::Scalar color1(0, 0, 255); 
    
    for (const auto& match : match_result) {
        int index = match.grid1;
        int i = index / num_cols1;
        int j = index % num_cols1;
        int grid_x = j * stride_w1;
        int grid_y = i * stride_h1;
        
        if (grid_x + grid_w1 <= frame_w1 && grid_y + grid_h1 <= frame_h1) {
            cv::Mat roi = result_frame1(cv::Rect(grid_x, grid_y, grid_w1, grid_h1));
            cv::Mat color_mask = cv::Mat::ones(roi.size(), roi.type());
            color_mask.setTo(color1);
            cv::addWeighted(roi, 1.0f - point, color_mask, point, 0, roi);
        }
    }
    
    int frame_w2 = result_frame2.cols;
    int frame_h2 = result_frame2.rows;
    int stride_w2 = stride2.width, stride_h2 = stride2.height;
    int grid_w2 = grid_size2.width, grid_h2 = grid_size2.height;
    int num_cols2 = (frame_w2 - grid_w2) / stride_w2 + 1;
    
    cv::Scalar color2(0, 255, 0); 
    
    for (const auto& match : match_result) {
        int index = match.grid2;
        int i = index / num_cols2;
        int j = index % num_cols2;
        int grid_x = j * stride_w2;
        int grid_y = i * stride_h2;
        
        if (grid_x + grid_w2 <= frame_w2 && grid_y + grid_h2 <= frame_h2) {
            cv::Mat roi = result_frame2(cv::Rect(grid_x, grid_y, grid_w2, grid_h2));
            cv::Mat color_mask = cv::Mat::ones(roi.size(), roi.type());
            color_mask.setTo(color2);
            cv::addWeighted(roi, 1.0f - point, color_mask, point, 0, roi);
        }
    }
    
    int target_height = std::min(result_frame1.rows, result_frame2.rows);
    if (result_frame1.rows != target_height) {
        cv::resize(result_frame1, result_frame1, cv::Size(result_frame1.cols * target_height / result_frame1.rows, target_height));
    }
    if (result_frame2.rows != target_height) {
        cv::resize(result_frame2, result_frame2, cv::Size(result_frame2.cols * target_height / result_frame2.rows, target_height));
    }
    
    cv::Mat combined_image;
    cv::hconcat(result_frame1, result_frame2, combined_image);
    
    std::string output_folder = output_path + "/" + file1 + "_" + file2;
    std::filesystem::create_directories(output_folder);
    std::string grid_size_str = std::to_string(grid_size1.width) + "x" + std::to_string(grid_size1.height);
    std::string output_file = output_folder + "/" + grid_size_str + ".jpg";
    
    bool success = cv::imwrite(output_file, combined_image);
    if (success) {
        std::cout << "Combined match result saved to " << output_file << std::endl;
    } else {
        std::cerr << "Error: Failed to save combined match result to " << output_file << std::endl;
    }
}

std::set<int> VideoMatcherUtils::getSmallIndexInLarge(int K, int num_large_grids_per_row, int num_small_grids_per_row,
                                                     int n_width, int n_height, bool shifting_flag) {
    int large_grid_row = K / num_large_grids_per_row;
    int large_grid_col = K % num_large_grids_per_row;
    
    int small_grid_row = large_grid_row * n_height;
    int small_grid_col = large_grid_col * n_width;
    
    if (shifting_flag) {
        n_height += 1;
        n_width += 1;
    }
    
    std::set<int> small_grid_set;
    
    std::vector<int> temp_indices;
    temp_indices.reserve(n_height * n_width);

    for (int i = 0; i < n_height; ++i) {
        for (int j = 0; j < n_width; ++j) {
            int small_grid_index = (small_grid_row + i) * num_small_grids_per_row + (small_grid_col + j);
            temp_indices.push_back(small_grid_index);
        }
    }
    
    small_grid_set.insert(temp_indices.begin(), temp_indices.end());
    
    return small_grid_set;
}

std::map<int, std::set<int>> VideoMatcherUtils::getSmallGridIndex(const std::vector<MatchTriplet>& match_result,
                                                                 const cv::Size& image_size, const cv::Size& small_grid_size,
                                                                 const cv::Size& large_grid_size, bool shifting_flag, int which_video) {
    if (which_video != 2) {
        return std::map<int, std::set<int>>();
    }
    
    int num_small_grids_per_row = image_size.width / small_grid_size.width;
    int num_large_grids_per_row = image_size.width / large_grid_size.width;
    
    if (shifting_flag) {
        num_small_grids_per_row = num_small_grids_per_row * 2 - 1;
        num_large_grids_per_row = num_large_grids_per_row * 2 - 1;
    }
    
    int n_width = large_grid_size.width / small_grid_size.width;
    int n_height = large_grid_size.height / small_grid_size.height;
    
    std::map<int, std::set<int>> large_grid_corre_small_dict;
    
    for (const auto& triplet : match_result) {
        int index1 = triplet.grid1;
        int index2 = triplet.grid2;
        
        std::set<int> small_grid_index_set = getSmallIndexInLarge(
            index2, num_large_grids_per_row, num_small_grids_per_row, n_width, n_height, shifting_flag);
        
        auto it = large_grid_corre_small_dict.find(index1);
        if (it != large_grid_corre_small_dict.end()) {
            it->second.insert(small_grid_index_set.begin(), small_grid_index_set.end());
        } else {
            large_grid_corre_small_dict[index1] = std::move(small_grid_index_set);
        }
    }
    
    return large_grid_corre_small_dict;
}

std::vector<MatchTriplet> VideoMatcherUtils::ransacFilterMatchResults(const std::vector<MatchTriplet>& match_results,
                                                                     const cv::Size& grid_size1, const cv::Size& grid_size2,
                                                                     int num_cols1, int num_rows1, int num_cols2, int num_rows2,
                                                                     bool shifting_flag, float ransac_threshold,
                                                                     int max_iterations, float min_inlier_ratio) {
    if (match_results.size() < 4) {
        std::cout << "匹配结果数量过少，跳过RANSAC筛选" << std::endl;
        return match_results;
    }
    
    std::cout << "开始RANSAC筛选，原始匹配数: " << match_results.size() << std::endl;
    
    std::vector<cv::Point2f> points1, points2;
    points1.reserve(match_results.size());
    points2.reserve(match_results.size());
    
    for (const auto& match : match_results) {
        int row1 = match.grid1 / num_cols1;
        int col1 = match.grid1 % num_cols1;
        int row2 = match.grid2 / num_cols2;
        int col2 = match.grid2 % num_cols2;
        float x1 = col1 * grid_size1.width + grid_size1.width * 0.5f;
        float y1 = row1 * grid_size1.height + grid_size1.height * 0.5f;
        float x2 = col2 * grid_size2.width + grid_size2.width * 0.5f;
        float y2 = row2 * grid_size2.height + grid_size2.height * 0.5f;
        
        points1.emplace_back(x1, y1);
        points2.emplace_back(x2, y2);
    }
    
    std::vector<uchar> inliers_mask;
    cv::Mat homography;
    
    try {
        homography = cv::findHomography(points1, points2, cv::RANSAC, 
                                       ransac_threshold, inliers_mask, max_iterations);
        
        if (homography.empty()) {
            std::cout << "无法计算单应性矩阵，返回原始匹配结果" << std::endl;
            return match_results;
        }
    } catch (const cv::Exception& e) {
        std::cout << "RANSAC计算异常: " << e.what() << "，返回原始匹配结果" << std::endl;
        return match_results;
    }
    
    int inlier_count = cv::sum(inliers_mask)[0];
    float inlier_ratio = static_cast<float>(inlier_count) / match_results.size();
    
    std::cout << "RANSAC内点数: " << inlier_count 
              << "，内点比例: " << inlier_ratio 
              << "，阈值: " << min_inlier_ratio << std::endl;

    if (inlier_ratio < min_inlier_ratio) {
        std::cout << "内点比例过低，返回原始匹配结果" << std::endl;
        return match_results;
    }
    
    std::vector<MatchTriplet> filtered_results;
    filtered_results.reserve(inlier_count);
    
    for (size_t i = 0; i < match_results.size(); ++i) {
        if (inliers_mask[i]) {
            filtered_results.push_back(match_results[i]);
        }
    }
    
    std::cout << "RANSAC筛选完成，保留匹配数: " << filtered_results.size() << std::endl;
    return filtered_results;
}

} // namespace VideoMatcher 