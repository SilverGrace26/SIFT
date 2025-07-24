#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "keypointDetection.hpp"

namespace kp {
  
bool isLocalExtremaPerOctave(const ss::Octave& DoG_octave, int scale_idx, int row, int col, const float contrast_threshold){

    const float val = DoG_octave[scale_idx].at<float>(row, col);
    if (std::abs(val) < contrast_threshold) return false;

    bool is_max = true;
    bool is_min = true;

    for (int ds = -1; ds <= 1; ds++) {
        const cv::Mat& neighbor_img = DoG_octave[scale_idx + ds];
        for (int dr = -1; dr <= 1; dr++) {
            for (int dc = -1; dc <= 1; dc++) {
                if (ds == 0 && dr == 0 && dc == 0) continue;

                int r = row + dr;
                int c = col + dc;
                if (r < 0 || r >= neighbor_img.rows || c < 0 || c >= neighbor_img.cols) continue;

                float neighbor_val = neighbor_img.at<float>(r, c);
                if (val <= neighbor_val) is_max = false;
                if (val >= neighbor_val) is_min = false;
            }
        }
    }

    return is_max || is_min;

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


