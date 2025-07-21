#pragma once 

#include <vector>
#include <opencv2/opencv.hpp>
#include "scaleSpace.hpp"

namespace dog {

  void validateOctaveImages(const ss::Octave& octave);
  void calculateDifferenceOfGaussiansPerOctave(const ss::Octave& octave, ss::Octave& single_DoG_octave);
  void calculateDifferenceOfGaussians(const ss::ScaleSpace& scale_space, ss::ScaleSpace& DoG_scale_space);
}

