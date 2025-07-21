#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

namespace ss {
  
  typedef std::vector<cv::Mat> Octave;
  typedef std::vector<Octave> ScaleSpace;

  float computeSigmaForLevel(float base_sigma, int level, int scalesPerOctave);
  float computeDeltaSigma(float prev_sigma, float curr_sigma);
  void prepareOctave(Octave& octave, cv::Mat base_image, float initial_scale, int scalesPerOctave);
  void prepareScaleSpace(ScaleSpace& scaleSpace, const cv::Mat& base_image, int numOctaves, int scalesPerOctave, float initial_scale);

}

