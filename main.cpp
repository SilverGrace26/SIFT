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


int main() {

    SIFT::Config config;

    // Load image
    cv::Mat original_input_8U = cv::imread("../Tower.jpeg", cv::IMREAD_GRAYSCALE); 
    if (original_input_8U.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return -1;
    }

    cv::Mat input;
    original_input_8U.convertTo(input, CV_32F);
    std::cout << "Image is Loaded." << '\n';

    //Execute SIFT API 
    std::pair<std::vector<Keypoint>, std::vector<desc::Desc>> result = executeSIFT(config, input_image);

    std::vector<kp::KeyPoint> refined_and_valid_keypoints = result.first;
    std::vector<desc::Desc> descriptors = result.second;

    std::cout << "Size of keypoints array after refinement : " << refined_and_valid_keypoints.size() << '\n';

    // Visualize keypoints - use the refined_and_valid_keypoints
    cv::Mat output_image_with_keypoints;
    vis::drawKeypoints(original_input_8U, refined_and_valid_keypoints, output_image_with_keypoints, cv::Scalar(0, 0, 255), 20); 

    showImage(output_image_with_keypoints, "Keypoints on Original Tower Image");

    return 0;
}
