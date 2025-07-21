#include <iostream>
#include <opencv2/opencv.hpp>
#include "sift.hpp" 

void showImage(cv::Mat image, const std::string& window_name = "Tower"){
  cv::Mat resized_image;
  cv::Mat display_image;

  if (image.type() == CV_32F) {
      image.convertTo(display_image, CV_8U); 
  } else {
      display_image = image;
  }

  cv::resize(display_image, resized_image, cv::Size(800, 800), cv::INTER_LINEAR);
  cv::imshow(window_name, resized_image);
  cv::waitKey(0);
}

std::vector<std::vector<float>> convertMatToVector(const cv::Mat& mat) {
    if (mat.empty() || mat.type() != CV_32F) {
        return {};
    }

    std::vector<std::vector<float>> vec2D(mat.rows, std::vector<float>(mat.cols));
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            vec2D[i][j] = mat.at<float>(i, j);
        }
    }
    return vec2D;
}

int main() {
    // Step 1: Load image
    cv::Mat original_input_8U = cv::imread("../Tower.jpeg", cv::IMREAD_GRAYSCALE); 
    if (original_input_8U.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }

    cv::Mat input;
    original_input_8U.convertTo(input, CV_32F);
    std::cout<<"Image is Loaded."<<'\n';

    int num_octaves = static_cast<int>(std::log2(std::min(input.rows, input.cols))) - 3;
    int scales_per_octave = 5;
    float initial_scale = 1.6;

    // Step 2: Build scale-space pyramid
    std::cout<<"Building Scale Space with Number of Octaves : "<<num_octaves<<'\n';
    SIFT::ScaleSpace scale_space;
    SIFT::prepareScaleSpace(scale_space, input, num_octaves, scales_per_octave, initial_scale);

    // Step 3: Compute Difference of Gaussian (DoG)
    std::cout<<"Building Difference of Gaussian Pyramid"<<'\n';
    SIFT::ScaleSpace DoG_pyramid;
    SIFT::calculateDifferenceOfGaussians(scale_space, DoG_pyramid);

    // Step 4 : Coarse Keypoint Detection 
    std::cout<<"Initiating Coarse Keypoint Detection"<<'\n';
    std::vector<SIFT::KeyPoint> keypoints;
    SIFT::coarseKeypointDetection(DoG_pyramid, keypoints, 0.04);

    // Converting CV::Mat to standard vector 
    std::vector<std::vector<float>> input_vec = convertMatToVector(input);

    std::vector<SIFT::KeyPoint> refined_and_valid_keypoints;

    int indicator = 0;

    for(auto& kp : keypoints){

      // Step 5: Refine keypoints (remove low contrast / edge responses)
      std::cout<<++indicator<<std::endl;
      SIFT::refineKeypoints(DoG_pyramid, kp);

      // Check if the keypoint is still valid after refinement
      if (kp.x != -1e6 && kp.y != -1e6) { 

          std::vector<std::vector<float>> histogram;
          SIFT::generateLogPolarHistogram(input_vec, kp, 8, 4, histogram);

          // Step 7: Compute descriptors
          desc::Desc descriptor = desc::createDescStruct(histogram);

          refined_and_valid_keypoints.push_back(kp);
      }
    }

    // Step 8: Visualize keypoints - use the refined_and_valid_keypoints
    cv::Mat output_image_with_keypoints;
    SIFT::drawKeypoints(original_input_8U, refined_and_valid_keypoints, output_image_with_keypoints, cv::Scalar(0, 0, 255), 20); 

    showImage(output_image_with_keypoints, "Keypoints on Original Tower Image");

    return 0;
}
