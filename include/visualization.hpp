#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "keypointDetection.hpp"

namespace vis {
void drawKeypoints(const cv::Mat& image, const std::vector<kp::KeyPoint>& keypoints, cv::Mat& output, const cv::Scalar& color, int line_thickness);

void drawKeypointsLines(const cv::Mat& image1, const std::vector<kp::KeyPoint>& keypoints1, const cv::Mat& image2, const std::vector<kp::KeyPoint>& keypoints2, cv::Mat& output, const cv::Scalar& color, int line_thickness);

} 
