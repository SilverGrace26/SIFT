#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include "dog.hpp"

// Helper function to create a test octave
ss::Octave createTestOctave(int levels, const cv::Size& size, int type) {
  ss::Octave octave;
  for (int i = 0; i < levels; ++i) {
      octave.push_back(cv::Mat::ones(size, type) * (i + 1));
  }
  return octave;
}

// Test validateOctaveImages with valid octave
TEST(ValidateOctaveImagesTest, ValidOctave) {
  ss::Octave octave = createTestOctave(3, cv::Size(10, 10), CV_32FC1);
  EXPECT_NO_THROW(dog::validateOctaveImages(octave));
}

// Test validateOctaveImages with empty octave
TEST(ValidateOctaveImagesTest, EmptyOctave) {
  ss::Octave octave;
  EXPECT_THROW(dog::validateOctaveImages(octave), std::invalid_argument);
}

// Test validateOctaveImages with inconsistent sizes
TEST(ValidateOctaveImagesTest, InconsistentSizes) {
  ss::Octave octave;
  octave.push_back(cv::Mat::ones(10, 10, CV_32FC1));
  octave.push_back(cv::Mat::ones(20, 20, CV_32FC1));
  EXPECT_THROW(dog::validateOctaveImages(octave), std::invalid_argument);
}

// Test validateOctaveImages with inconsistent types
TEST(ValidateOctaveImagesTest, InconsistentTypes) {
  ss::Octave octave;
  octave.push_back(cv::Mat::ones(10, 10, CV_32FC1));
  octave.push_back(cv::Mat::ones(10, 10, CV_8UC1));
  EXPECT_THROW(dog::validateOctaveImages(octave), std::invalid_argument);
}

// Test calculateDifferenceOfGaussiansPerOctave with valid input
TEST(CalculateDoGPerOctaveTest, ValidInput) {
  ss::Octave octave = createTestOctave(3, cv::Size(10, 10), CV_32FC1);
  ss::Octave DoG_octave;
    
  EXPECT_NO_THROW(dog::calculateDifferenceOfGaussiansPerOctave(octave, DoG_octave));
  EXPECT_EQ(DoG_octave.size(), 2);
  
  // Check the difference values
  for (const auto& img : DoG_octave) {
      EXPECT_TRUE(cv::countNonZero(img == 1.0f) == 100);
  }
}

// Test calculateDifferenceOfGaussiansPerOctave with insufficient images
TEST(CalculateDoGPerOctaveTest, InsufficientImages) {
  ss::Octave octave = createTestOctave(1, cv::Size(10, 10), CV_32FC1);
  ss::Octave DoG_Octave;
  EXPECT_THROW(dog::calculateDifferenceOfGaussiansPerOctave(octave, DoG_Octave), std::invalid_argument);
}

// Test calculateDifferenceOfGaussians with valid scale space
TEST(CalculatedogTest, ValidScaleSpace) {
  ss::ScaleSpace scale_space(2);
  scale_space[0] = createTestOctave(3, cv::Size(10, 10), CV_32FC1);
  scale_space[1] = createTestOctave(4, cv::Size(5, 5), CV_32FC1);
    
  ss::ScaleSpace DoG_scale_space;
  EXPECT_NO_THROW(dog::calculateDifferenceOfGaussians(scale_space, DoG_scale_space));
    
  EXPECT_EQ(DoG_scale_space.size(), 2);
  EXPECT_EQ(DoG_scale_space[0].size(), 2);  // 3-1=2 differences
  EXPECT_EQ(DoG_scale_space[1].size(), 3);  // 4-1=3 differences
}

// Test calculateDifferenceOfGaussians with empty scale space
TEST(CalculatedogTest, EmptyScaleSpace) {
  ss::ScaleSpace scale_space;
  ss::ScaleSpace DoG_scale_space;
  EXPECT_THROW(dog::calculateDifferenceOfGaussians(scale_space, DoG_scale_space), std::invalid_argument);
}

// Test calculateDifferenceOfGaussians with empty octave
TEST(CalculatedogTest, EmptyOctaveInScaleSpace) {
  ss::ScaleSpace scale_space(2);
  scale_space[0] = createTestOctave(3, cv::Size(10, 10), CV_32FC1);
  scale_space[1] = ss::Octave();  // Empty octave
    
  ss::ScaleSpace DoG_scale_space;
  EXPECT_THROW(dog::calculateDifferenceOfGaussians(scale_space, DoG_scale_space), std::invalid_argument);
}

// Test the actual difference calculation
TEST(dogCalculationTest, CorrectDifference) {
  ss::Octave octave;
  cv::Mat img1 = (cv::Mat_<float>(2, 2) << 1.0, 2.0, 3.0, 4.0);
  cv::Mat img2 = (cv::Mat_<float>(2, 2) << 2.0, 3.0, 4.0, 5.0);
  octave.push_back(img1);
  octave.push_back(img2);
    
  ss::Octave DoG_octave;
  dog::calculateDifferenceOfGaussiansPerOctave(octave, DoG_octave);
    
  cv::Mat expected = (cv::Mat_<float>(2, 2) << 1.0, 1.0, 1.0, 1.0);
  EXPECT_TRUE(cv::countNonZero(DoG_octave[0] != expected) == 0);
}
