#pragma once

#include <vector>
#include <complex>
#include "scaleSpace.hpp"

namespace desc {
  
struct Match {
int idx1;
int idx2;
float distance;
};

struct Desc {
std::vector<std::complex<float>> descriptor;
float dominant_orientation;
};

std::vector<float> flattenHist(const std::vector<std::vector<float>>& hist);
std::vector<std::complex<float>> calculateDFT(const std::vector<float>& descriptor);
float findDominantOrientation(const std::vector<float>& histogram);
void l2Normalize(std::vector<float>& desc);
void l2Normalize(std::vector<std::complex<float>>& desc); 
Desc createDescStruct(const std::vector<std::vector<float>> & histogram);
float euclideanDistance(const std::vector<std::complex<float>>& a, const std::vector<std::complex<float>>& b);
std::vector<desc::Match> matchDescriptorSets(const std::vector<desc::Desc>& set1, const std::vector<desc::Desc>& set2, float ratio = 0.8f); 

}
