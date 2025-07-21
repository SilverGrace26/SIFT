#include <iostream>
#include <complex>
#include <numbers>
#include <vector>
#include <numeric>
#include <algorithm>
#include <limits>
#include "descriptor.hpp"

namespace desc {


std::vector<float> flattenHist(const std::vector<std::vector<float>>& hist) {
    std::vector<float> descriptor;
    for (const auto& angle_bin : hist) {
        descriptor.insert(descriptor.end(), angle_bin.begin(), angle_bin.end());
    }
    return descriptor;
}

std::vector<std::complex<float>> calculateDFT(const std::vector<float>& descriptor) {
    int N = descriptor.size();
    std::vector<std::complex<float>> dft(N);

    for (int k = 0; k < N; ++k) {
        std::complex<float> sum(0.0f, 0.0f);
        for (int n = 0; n < N; ++n) {
            float angle = (-2.0f * std::numbers::pi_v<float> * k * n) / N;
            sum += std::polar(descriptor[n], angle);
        }
        dft[k] = sum;
    }

    return dft;
}

float findDominantOrientation(const std::vector<float>& histogram) {
    int dominant_bin = std::max_element(histogram.begin(), histogram.end()) - histogram.begin();
    return (360.0f / histogram.size()) * dominant_bin;
}

// --- Overloaded L2 Normalization ---

void l2Normalize(std::vector<float>& desc) {
    float norm = std::sqrt(std::inner_product(desc.begin(), desc.end(), desc.begin(), 0.0f));
    if (norm > 1e-6f) {
        for (auto& val : desc) val /= norm;
    }
}

void l2Normalize(std::vector<std::complex<float>>& desc) {
    float norm = 0.0f;
    for (const auto& val : desc) {
        norm += std::norm(val);  
    }
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (auto& val : desc) val /= norm;
    }
}


Desc createDescStruct(const std::vector<std::vector<float>>& histogram) {
    std::vector<float> descriptor = flattenHist(histogram);
    float dominant_orientation = findDominantOrientation(descriptor);
    l2Normalize(descriptor);
    std::vector<std::complex<float>> DFT_descriptor = calculateDFT(descriptor);
    l2Normalize(DFT_descriptor);

    return Desc{DFT_descriptor, dominant_orientation};
}


float euclideanDistance(const std::vector<std::complex<float>>& a, const std::vector<std::complex<float>>& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        std::complex<float> diff = a[i] - b[i];
        sum += std::norm(diff);  
    }
    return std::sqrt(sum);
}


std::vector<Match> matchDescriptorSets(const std::vector<Desc>& set1, const std::vector<Desc>& set2, float ratio) {
    std::vector<Match> matches;

    for (size_t i = 0; i < set1.size(); ++i) {
        float best_dist = std::numeric_limits<float>::max();
        float second_best_dist = std::numeric_limits<float>::max();
        int best_j = -1;

        for (size_t j = 0; j < set2.size(); ++j) {
            float dist = euclideanDistance(set1[i].descriptor, set2[j].descriptor);

            if (dist < best_dist) {
                second_best_dist = best_dist;
                best_dist = dist;
                best_j = j;
            } else if (dist < second_best_dist) {
                second_best_dist = dist;
            }
        }

        if (best_j != -1 && best_dist < ratio * second_best_dist) {
            matches.push_back({static_cast<int>(i), best_j, best_dist});
        }
    }

    return matches;
}

}  // namespace desc
