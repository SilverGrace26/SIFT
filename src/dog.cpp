#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include "dog.hpp"

namespace dog {

void validateOctaveImages(const ss::Octave& octave) {
  if (octave.empty()) {
    throw std::invalid_argument("Octave must contain at least one image.");
  }

  cv::Size size = octave[0].size();
  int type = octave[0].type();

  for (size_t i = 1; i < octave.size(); ++i) {
    if (octave[i].size() != size) {
      throw std::invalid_argument("All images in an octave must have the same dimensions.");
    }
    if (octave[i].type() != type) {
      throw std::invalid_argument("All images in an octave must have the same type.");
    }
  }
}

void calculateDifferenceOfGaussiansPerOctave(const ss::Octave& octave, ss::Octave& single_DoG_octave) {
  if (octave.size() < 2) {
    throw std::invalid_argument("Number of images per octave should be at least 2.");
  }

  validateOctaveImages(octave);

  single_DoG_octave.resize(octave.size() - 1);

  for (size_t image_idx = 1; image_idx < octave.size(); image_idx++) {
    if (octave[image_idx].size() != octave[image_idx - 1].size() ||
        octave[image_idx].type() != octave[image_idx - 1].type()) {
      throw std::invalid_argument("Consecutive images must have the same size and type.");
    }

    single_DoG_octave[image_idx - 1] = octave[image_idx] - octave[image_idx - 1];
  }
}

void calculateDifferenceOfGaussians(const ss::ScaleSpace& scale_space, ss::ScaleSpace& DoG_scale_space) {
  if (scale_space.empty()) {
    throw std::invalid_argument("Scale Space should not be empty.");
  }

  DoG_scale_space.resize(scale_space.size());

  for (size_t octave_idx = 0; octave_idx < scale_space.size(); octave_idx++) {
    const ss::Octave& octave = scale_space[octave_idx];

    if (octave.empty()) {
      throw std::invalid_argument("Octave " + std::to_string(octave_idx) + " is empty.");
    }

    calculateDifferenceOfGaussiansPerOctave(octave, DoG_scale_space[octave_idx]);
  }
}

} 
