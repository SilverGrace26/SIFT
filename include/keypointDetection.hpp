#pragma once 

#include <vector>
#include <opencv2/opencv.hpp>
#include "scaleSpace.hpp"

namespace kp {
  
  struct KeyPoint {
      float x;              
      float y;             
      float scale_idx;    
      int octave_idx;  
      float DoG_value;
  };

  bool isLocalExtremaPerOctave(const ss::Octave& DoG_octave, int scale_idx, int row, int col, float contrast_threshold);

  void coarseKeypointDetection(const ss::ScaleSpace& DoG_scale_space, std::vector<KeyPoint>& keypoints, float contrast_threshold);

}


