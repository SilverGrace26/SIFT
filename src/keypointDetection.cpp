#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "keypointDetection.hpp"

namespace kp {
  
bool isLocalExtremaPerOctave(const ss::Octave& DoG_octave, int scale_idx, int row, int col, const float contrast_threshold){

  const float val = DoG_octave[scale_idx].at<float>(row, col);
  if(val < contrast_threshold) return false;
      
  for(int d_scale_idx = -1 ; d_scale_idx <= 1; d_scale_idx++){
    
    const cv::Mat image = DoG_octave[scale_idx + d_scale_idx];

    for(int d_row = -1; d_row <= 1; d_row++){
      for(int d_col = -1; d_col <= 1; d_col++){

        if (d_scale_idx == 0 && d_row == 0 && d_col == 0) continue;     

        // For borders
        if (row + d_row < 0 || row + d_row >= image.rows || col + d_col < 0 || col + d_col >= image.cols) continue;

        if (val <= image.at<float>(row + d_row, col + d_col)) return false;
      }
    }

  }

  return true;

}
  
void coarseKeypointDetection(const ss::ScaleSpace& DoG_scale_space, std::vector<KeyPoint>& keypoints, const float contrast_threshold){

  for(int octave_idx = 0; octave_idx < DoG_scale_space.size(); octave_idx++){
    for(int scale_idx = 1 ; scale_idx < DoG_scale_space[octave_idx].size() - 1; scale_idx++){

      const cv::Mat& image = DoG_scale_space[octave_idx][scale_idx];

      for(int row = 1; row < image.rows - 1; row++){
        for(int col = 1; col < image.cols - 1; col++){
          if ( isLocalExtremaPerOctave(DoG_scale_space[octave_idx], scale_idx, row, col, contrast_threshold) ) {

            int x = col * (1 << octave_idx);
            int y = row * (1 << octave_idx);
            float val = image.at<float>(row, col);
            keypoints.push_back({x, y, scale_idx, octave_idx, val});
          }

        }
      }

    }
  }

}


}


