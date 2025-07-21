#include <iostream>
#include <vector>
#include <numbers>
#include <utility>
#include <cmath>
#include <algorithm> 
#include "histogram.hpp"

namespace hist {
  
std::vector<std::pair<int, int>> generateCircularMask(float radius) {
    std::vector<std::pair<int, int>> mask;
    int r = static_cast<int>(std::ceil(radius));

    for (int d_row = -r; d_row <= r; d_row++) {
        for (int d_col = -r; d_col <= r; d_col++) {
            if (d_row * d_row + d_col * d_col <= radius * radius) {
                mask.emplace_back(d_row, d_col);
            }
        }
    }

    return mask;
}

std::pair<float, float> calculateAngleAndLogRadius(int dx, int dy, float max_radius) {
    float angle = std::atan2(dy, dx);
    if (angle < 0) angle += 2 * std::numbers::pi_v<float>;
    float angle_degrees = angle * (180.0f / std::numbers::pi_v<float>);

    float radius = std::sqrt(dx*dx + dy*dy);
    float log_radius = std::log(radius + 1e-6f);

    return std::make_pair(angle_degrees, log_radius);
}

float calculateGaussianWindowWeight(int dx, int dy, float sigma) {
    float sigma_window = 1.5f * sigma;
    float dist_sq = dx * dx + dy * dy;
    return std::exp(-dist_sq / (2.0f * sigma_window * sigma_window));
}

void generateLogPolarHistogram(const std::vector<std::vector<float>>& image, const kp::KeyPoint& keypoint,
    int num_angle_bins, int num_radius_bins, std::vector<std::vector<float>>& histogram) {
    int image_height = image.size();
    int image_width = image_height > 0 ? image[0].size() : 0;

    histogram.assign(num_angle_bins, std::vector<float>(num_radius_bins, 0.0f));

    int x0_int = static_cast<int>(std::round(keypoint.x));
    int y0_int = static_cast<int>(std::round(keypoint.y));

    float sigma = keypoint.scale_idx; 

    float max_radius = 4.5f * sigma;
    if (max_radius <= 1e-6f) max_radius = 1e-6f;
    float log_max_radius = std::log(max_radius);

    auto offsets = generateCircularMask(max_radius);

    for (const auto& [dx, dy] : offsets) {
        int x = x0_int + dx;
        int y = y0_int + dy;

        // Boundary check for the 'image' (input_vec)
        if (x < 0 || x >= image_width || y < 0 || y >= image_height)
            continue; 

        float intensity = image[y][x];

        float angle_degrees;
        float log_radius;
        if (dx == 0 && dy == 0) {
            angle_degrees = 0.0f; 
            log_radius = std::log(1e-6f); 
        } else {
            std::tie(angle_degrees, log_radius) = calculateAngleAndLogRadius(dx, dy, max_radius);
        }

        // Normalizing log_radius to the range [0, 1] for binning
        float normalized_log_radius = 0.0f;
        if (log_max_radius > 1e-6f) {
            normalized_log_radius = log_radius / log_max_radius;
        }

        int angle_bin = static_cast<int>((angle_degrees / 360.0f) * num_angle_bins);
        angle_bin = (angle_bin % num_angle_bins + num_angle_bins) % num_angle_bins; 

        int radius_bin = static_cast<int>(normalized_log_radius * (num_radius_bins)); 
        radius_bin = std::clamp(radius_bin, 0, num_radius_bins - 1); 

        float weight = calculateGaussianWindowWeight(dx, dy, sigma);

        histogram[angle_bin][radius_bin] += weight * intensity;
    }
}

}
