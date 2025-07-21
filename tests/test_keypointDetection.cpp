#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "keypointDetection.hpp"


ss::Octave createTestOctave(float center_val, float surrounding_val) {
    ss::Octave octave;

    // Create 3 layers with surrounding_val
    for (int i = 0; i < 3; ++i) {
        octave.push_back(cv::Mat::ones(5, 5, CV_32F) * surrounding_val);
    }

    // Set only the center pixel of the middle scale to center_val
    octave[1].at<float>(2, 2) = center_val;

    return octave;
}

TEST(IsLocalExtremaPerOctaveTest, ReturnsTrueWhenMaximum) {
    float center_val = 10.0f;
    float surrounding_val = 1.0f;
    float threshold = 5.0f;

    ss::Octave octave = createTestOctave(center_val, surrounding_val);
    EXPECT_TRUE(kp::isLocalExtremaPerOctave(octave, 1, 2, 2, threshold));
}

TEST(IsLocalExtremaPerOctaveTest, ReturnsFalseWhenBelowThreshold) {
    float center_val = 2.0f;
    float surrounding_val = 1.0f;
    float threshold = 5.0f;

    ss::Octave octave = createTestOctave(center_val, surrounding_val);
    EXPECT_FALSE(kp::isLocalExtremaPerOctave(octave, 1, 2, 2, threshold));
}

TEST(IsLocalExtremaPerOctaveTest, ReturnsFalseWhenNotMaximum) {
    float center_val = 5.0f;
    float surrounding_val = 10.0f;
    float threshold = 1.0f;

    ss::Octave octave = createTestOctave(center_val, surrounding_val);
    EXPECT_FALSE(kp::isLocalExtremaPerOctave(octave, 1, 2, 2, threshold));
}

TEST(CoarseKeypointDetectionTest, DetectsKeypointAtExpectedLocation) {
    float center_val = 10.0f;
    float surrounding_val = 1.0f;
    float threshold = 5.0f;

    ss::Octave octave = createTestOctave(center_val, surrounding_val);
    ss::ScaleSpace scale_space = {octave};

    std::vector<kp::KeyPoint> keypoints;
    kp::coarseKeypointDetection(scale_space, keypoints, threshold);

    ASSERT_EQ(keypoints.size(), 1);
    EXPECT_EQ(keypoints[0].x, 2 * (1 << 0));
    EXPECT_EQ(keypoints[0].y, 2 * (1 << 0));
    EXPECT_EQ(keypoints[0].scale_idx, 1);
    EXPECT_EQ(keypoints[0].octave_idx, 0);
    EXPECT_FLOAT_EQ(keypoints[0].DoG_value, center_val);
}

TEST(CoarseKeypointDetectionTest, NoKeypointsWhenBelowThreshold) {
    float center_val = 1.0f;
    float surrounding_val = 1.0f;
    float threshold = 5.0f;

    ss::Octave octave = createTestOctave(center_val, surrounding_val);
    ss::ScaleSpace scale_space = {octave};

    std::vector<kp::KeyPoint> keypoints;
    kp::coarseKeypointDetection(scale_space, keypoints, threshold);

    EXPECT_TRUE(keypoints.empty());
}
