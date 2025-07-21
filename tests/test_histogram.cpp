#include <gtest/gtest.h>
#include <cmath>
#include <numbers>
#include <vector>
#include "histogram.hpp"

TEST(HistogramUtilsTest, GeneratesCorrectCircularMask) {
    auto mask = hist::generateCircularMask(1.5f);
    std::set<std::pair<int, int>> expected = {
        {-1, -1}, {-1, 0}, {-1, 1},
        {0, -1}, {0, 0}, {0, 1},
        {1, -1}, {1, 0}, {1, 1}
    };

    for (const auto& p : expected) {
        EXPECT_TRUE(std::find(mask.begin(), mask.end(), p) != mask.end());
    }

    // Ensure no point outside radius is present
    for (const auto& [dx, dy] : mask) {
        EXPECT_LE(dx*dx + dy*dy, 1.5f * 1.5f + 1e-5f);
    }
}

TEST(HistogramUtilsTest, CalculatesCorrectAngleAndRadius) {
    auto [angle_deg, log_radius] = hist::calculateAngleAndLogRadius(1, 1, 10.0f);

    float expected_angle = 45.0f;
    float expected_radius = std::log(std::sqrt(2.0f) + 1e-6f);

    EXPECT_NEAR(angle_deg, expected_angle, 1e-2f);
    EXPECT_NEAR(log_radius, expected_radius, 1e-5f);
}

TEST(HistogramUtilsTest, CalculatesCorrectGaussianWeight) {
    float sigma = 1.0f;
    float weight_center = hist::calculateGaussianWindowWeight(0, 0, sigma);
    float weight_offset = hist::calculateGaussianWindowWeight(1, 0, sigma);

    EXPECT_GT(weight_center, weight_offset);
    EXPECT_NEAR(weight_center, 1.0f, 1e-5f);  // Gaussian at center is ~1
}

TEST(HistogramUtilsTest, GeneratesReasonableHistogram) {
    std::vector<std::vector<float>> image(5, std::vector<float>(5, 0.0f));
    image[2][2] = 1.0f;
    image[2][3] = 0.5f;
    image[1][2] = 0.25f;

    kp::KeyPoint kp = {2.0f, 2.0f, 1.0f, 0};  // x, y, scale, octave

    int angle_bins = 8;
    int radius_bins = 4;
    std::vector<std::vector<float>> hist;

    hist::generateLogPolarHistogram(image, kp, angle_bins, radius_bins, hist);

    float total = 0.0f;
    for (auto& row : hist)
        for (float val : row)
            total += val;

    EXPECT_GT(total, 1.0f);  // Should have non-zero contribution
    EXPECT_EQ(hist.size(), angle_bins);
    EXPECT_EQ(hist[0].size(), radius_bins);
}
