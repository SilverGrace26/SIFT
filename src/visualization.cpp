#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include "visualization.hpp"

namespace vis {
  
void drawKeypoints(const cv::Mat& image, const std::vector<kp::KeyPoint>& keypoints, 
    cv::Mat& output, const cv::Scalar& color, int line_thickness) {
    if(image.empty()){
      throw std::invalid_argument("Image should not be empty.");
    }

    if (image.channels() == 1)
        cv::cvtColor(image, output, cv::COLOR_GRAY2BGR);
    else
        output = image.clone();
    
    if(output.empty()){
      throw std::logic_error("Output empty due to some reason.");
    }

    for (const auto& kp : keypoints) {
      int x = static_cast<int>(kp.x);
      int y = static_cast<int>(kp.y);
      cv::circle(output, {x, y}, 2, color, line_thickness);
    }
}


void drawKeypointsLines(const cv::Mat& image1, const std::vector<kp::KeyPoint>& keypoints1, const cv::Mat& image2, const std::vector<kp::KeyPoint>& keypoints2, cv::Mat& output, const cv::Scalar& color, int line_thickness) {

    output = image1.clone();
    for (int i = 0; i < std::min(keypoints1.size(), keypoints2.size()); ++i) {
        if (keypoints1[i].x != -1e6 && keypoints1[i].y != -1e6 && keypoints2[i].x != -1e6 && keypoints2[i].y != -1e6) {
            cv::line(output, {static_cast<int>(keypoints1[i].x), static_cast<int>(keypoints1[i].y)}, {static_cast<int>(keypoints2[i].x), static_cast<int>(keypoints2[i].y)}, color, line_thickness);
        }
    }
}

}
