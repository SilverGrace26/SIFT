#include <iostream>
#include <vector>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include "scaleSpace.hpp"

namespace ss {
  
typedef std::vector<cv::Mat> Octave;
typedef std::vector<Octave> ScaleSpace;

float computeSigmaForLevel(float base_sigma, int level, int scalesPerOctave) {
    return base_sigma * std::pow(2.0, float(level) / scalesPerOctave); 
}

float computeDeltaSigma(float prev_sigma, float curr_sigma) {
    if (curr_sigma < prev_sigma){
      throw std::domain_error("Sigma values must be non-decreasing");
    }
    return std::sqrt(curr_sigma * curr_sigma - prev_sigma * prev_sigma); 
}

void prepareOctave(Octave& octave, cv::Mat base_image, float initial_scale, int scalesPerOctave) {

    octave.resize(scalesPerOctave + 2);

    std::vector<float> sigmas(scalesPerOctave + 2);
    for (int i = 0; i < scalesPerOctave + 2; ++i) {
        sigmas[i] = computeSigmaForLevel(initial_scale, i, scalesPerOctave);
    }

    octave[0] = base_image;

    for (int image_idx = 1; image_idx < scalesPerOctave + 2; image_idx++) {

        if(octave[image_idx - 1].empty()){
          throw std::logic_error("Previous image is unexpectedly empty.");
        }

        float delta_sigma = computeDeltaSigma(sigmas[image_idx - 1], sigmas[image_idx]);
        cv::GaussianBlur(octave[image_idx - 1], octave[image_idx], cv::Size(0, 0), delta_sigma, delta_sigma, cv::BORDER_REFLECT101);

    }
}

void prepareScaleSpace(ScaleSpace& scaleSpace, const cv::Mat& base_image, int numOctaves, int scalesPerOctave, float initial_scale) {

    if (base_image.empty() || base_image.type() != CV_32F) {
        throw std::invalid_argument("Base image must be non-empty and of type CV_32F.");
    }

    if (numOctaves <= 0 || scalesPerOctave <= 0) {
        throw std::invalid_argument("Number of octaves and scales per octave must be positive.");
    } 

    if(initial_scale <= 0){
      throw std::invalid_argument("Initial Scale must be greater than 0.");
    }
    
    scaleSpace.resize(numOctaves);

    prepareOctave(scaleSpace[0], base_image.clone(), initial_scale, scalesPerOctave);

    for (int octave = 1; octave < numOctaves; octave++) {

        cv::Mat base_image_downsampled;
        cv::Mat downsampling_image = scaleSpace[octave - 1].back();

          if (downsampling_image.empty() || downsampling_image.cols == 0 || downsampling_image.rows == 0) {
              throw std::logic_error("Downsampling image is invalid (empty or zero-size) before resize");
          }
        cv::resize(downsampling_image, base_image_downsampled, cv::Size(), 0.5, 0.5);
          
        prepareOctave(scaleSpace[octave], base_image_downsampled, initial_scale, scalesPerOctave);
    }
}

}

