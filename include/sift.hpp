#pragma once

#include "scaleSpace.hpp"
#include "dog.hpp"
#include "keypointDetection.hpp"
#include "refine.hpp"
#include "histogram.hpp"
#include "descriptor.hpp"
#include "visualization.hpp"

namespace SIFT {
  
  struct Config{
    int scales_per_octave = 5;
    float initial_scale = 1.6;
    float contrast_threshold = 0.04f;
    float edge_threshold = 10.0f;
    float offset_limit = 1.0f;
    int num_angular_bins = 8;
    int num_radial_bins = 4;
  };

  std::vector<kp::KeyPoint> executeSIFT(const Config& config, cv::Mat& input);

}
