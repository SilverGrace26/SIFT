#pragma once

#include <vector>
#include <functional>
#include <opencv2/opencv.hpp>
#include "scaleSpace.hpp"
#include "keypointDetection.hpp"

namespace refine {

  float calculateDeterminant(std::vector<std::vector<float>>& Hessian);

  std::vector<std::vector<float>> calculateAdjugate(std::vector<std::vector<float>>& Hessian);

  std::vector<std::vector<float>> calculateInverse(std::vector<std::vector<float>>& Hessian);

  float accessScaleSpace(const ss::ScaleSpace& DoG_scale_space, const kp::KeyPoint& keypoint);

  std::vector<float> calculateKeypointGradients(const ss::ScaleSpace& DoG_scale_space, const kp::KeyPoint& keypoint);

  std::vector<std::vector<float>> calculateKeypointHessian(const ss::ScaleSpace& DoG_scale_space, const kp::KeyPoint& keypoint);

  bool isOnEdge(const std::vector<std::vector<float>>& hessian, float edge_threshold);

  void refineKeypoints(const ss::ScaleSpace& DoG_scale_space, kp::KeyPoint& keypoint);

}

