#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "scaleSpace.hpp"

// ---------- Basic tests ----------
TEST(SigmaTest, ComputeSigmaForLevel) {
    float sigma = ss::computeSigmaForLevel(1.6, 2, 3);
    EXPECT_NEAR(sigma, 1.6 * std::pow(2.0, 2.0 / 3.0), 1e-5);
}

TEST(SigmaTest, ComputeDeltaSigmaValid) {
    float s1 = 1.6;
    float s2 = 2.0;
    float delta = ss::computeDeltaSigma(s1, s2);
    EXPECT_NEAR(delta, std::sqrt(s2*s2 - s1*s1), 1e-5);
}

TEST(SigmaTest, ComputeDeltaSigmaInvalid) {
    EXPECT_THROW(ss::computeDeltaSigma(2.0, 1.0), std::domain_error);
}

TEST(ScaleSpaceTest, PrepareScaleSpaceImageCheck) {
    cv::Mat image = cv::Mat::ones(64, 64, CV_32F);
    ss::ScaleSpace scale_space;
    ss::prepareScaleSpace(scale_space, image, 3, 3, 1.6);
    EXPECT_EQ(scale_space.size(), 3); 
    for (auto& octave : scale_space) {
        EXPECT_EQ(octave.size(), 5); // 3 + 2 = 5 images
    }
}
