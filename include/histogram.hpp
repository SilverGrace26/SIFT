#pragma once

#include <vector>
#include <utility> 
#include "keypointDetection.hpp"

namespace hist {
  
std::vector<std::pair<int, int>> generateCircularMask(float radius);

std::pair<float, float> calculateAngleAndLogRadius(int dx, int dy, float max_radius);

float calculateGaussianWindowWeight(int dx, int dy, float sigma);

void generateLogPolarHistogram(const std::vector<std::vector<float>>& image, const kp::KeyPoint& keypoint,
    int num_angle_bins, int num_radius_bins, std::vector<std::vector<float>>& histogram);

}

