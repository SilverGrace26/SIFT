#include <iostream>
#include <opencv2/opencv.hpp>
#include "sift.hpp" 

namespace SIFT {
  
static std::vector<std::vector<float>> convertMatToVector(const cv::Mat& mat) {
    if (mat.empty() || mat.type() != CV_32F) return {};

    std::vector<std::vector<float>> vec2D(mat.rows, std::vector<float>(mat.cols));
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) 
          vec2D[i][j] = mat.at<float>(i, j);
    }

    return vec2D;
}

std::vector<kp::KeyPoint> executeSIFT(const Config& config, cv::Mat& input){

    int scales_per_octave = config.scales_per_octave;
    float initial_scale = config.initial_scale;
    float contrast_threshold = config.contrast_threshold;
    float edge_threshold = config.edge_threshold;
    float offset_limit = config.offset_limit;
    int num_angular_bins = config.num_angular_bins;
    int num_radial_bins = config.num_radial_bins;
    
    int num_octaves = static_cast<int>(std::log2(std::min(input.rows, input.cols))) - 3;

    // Step 2: Build scale-space pyramid
    std::cout<<"Building Scale Space with Number of Octaves : "<<num_octaves<<'\n';
    ss::ScaleSpace scale_space;
    ss::prepareScaleSpace(scale_space, input, num_octaves, scales_per_octave, initial_scale);

    // Step 3: Compute Difference of Gaussian (DoG)
    std::cout<<"Building Difference of Gaussian Pyramid"<<'\n';
    ss::ScaleSpace DoG_pyramid;
    dog::calculateDifferenceOfGaussians(scale_space, DoG_pyramid);

    // Step 4 : Coarse Keypoint Detection 
    std::cout<<"Initiating Coarse Keypoint Detection"<<'\n';
    std::vector<kp::KeyPoint> keypoints;
    kp::coarseKeypointDetection(DoG_pyramid, keypoints, contrast_threshold);

    // Converting CV::Mat to standard vector 
    std::vector<std::vector<float>> input_vec = convertMatToVector(input);

    std::cout<<"Initial size of kepoints array : "<<keypoints.size()<<'\n';
    std::vector<kp::KeyPoint> refined_and_valid_keypoints;

    for(auto& kp : keypoints){

      // Step 5: Refine keypoints (remove low contrast / edge responses)
      refine::refineKeypoints(DoG_pyramid, kp);

      // Check if the keypoint is still valid after refinement
      if (kp.x != -1e6 && kp.y != -1e6) { 

          std::vector<std::vector<float>> histogram;
          hist::generateLogPolarHistogram(input_vec, kp, num_angular_bins, num_radial_bins, histogram);

          // Step 7: Compute descriptors
          desc::Desc descriptor = desc::createDescStruct(histogram);

          refined_and_valid_keypoints.push_back(kp);
      }
    }

    return refined_and_valid_keypoints;
}

}
