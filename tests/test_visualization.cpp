#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include "visualization.hpp"

TEST(VisUtils, DrawKeypoints) {
    cv::Mat image = cv::Mat::zeros(100, 100, CV_8UC3);

    std::vector<kp::KeyPoint> keypoints;
    for (int i = 0; i < 5; ++i) {
        keypoints.push_back({float(i * 20), float(i * 20), 0.0f, 0, 0.0f});
    }

    cv::Mat output;
    vis::drawKeypoints(image, keypoints, output, cv::Scalar(0, 255, 0), 2);

    EXPECT_FALSE(output.empty());
}

TEST(VisUtils, DrawKeypointsLines) {
    cv::Mat image1 = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::Mat image2 = cv::Mat::zeros(100, 100, CV_8UC3);

    std::vector<kp::KeyPoint> keypoints1, keypoints2;
    for (int i = 0; i < 5; ++i) {
        keypoints1.push_back({float(i * 20), float(i * 20), 0.0f, 0, 0.0f});
        keypoints2.push_back({float(i * 30), float(i * 30), 0.0f, 0, 0.0f});
    }

    cv::Mat output;
    vis::drawKeypointsLines(image1, keypoints1, image2, keypoints2, output, cv::Scalar(255, 0, 0), 2);

    cv::Mat diff;
    cv::absdiff(output, image1, diff);
    cv::Mat gray_diff;
    cv::cvtColor(diff, gray_diff, cv::COLOR_BGR2GRAY);

    EXPECT_FALSE(output.empty());
    EXPECT_GT(cv::countNonZero(gray_diff), 0);
}
