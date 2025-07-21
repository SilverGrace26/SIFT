#include <gtest/gtest.h>
#include <complex>
#include "descriptor.hpp"

using namespace desc;

TEST(FlattenHistTest, Flattens2DHistogramCorrectly) {
    std::vector<std::vector<float>> hist = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };
    std::vector<float> flat = flattenHist(hist);
    ASSERT_EQ(flat.size(), 4);
    EXPECT_EQ(flat[0], 1.0f);
    EXPECT_EQ(flat[1], 2.0f);
    EXPECT_EQ(flat[2], 3.0f);
    EXPECT_EQ(flat[3], 4.0f);
}

TEST(RealNormalizationTest, NormalizesFloatDescriptorL2) {
    std::vector<float> vec = {3.0f, 4.0f};
    l2Normalize(vec);
    float norm = std::sqrt(vec[0]*vec[0] + vec[1]*vec[1]);
    EXPECT_NEAR(norm, 1.0f, 1e-5f);
}

TEST(ComplexNormalizationTest, NormalizesComplexDescriptorL2) {
    std::vector<std::complex<float>> vec = {
        {3.0f, 0.0f},
        {0.0f, 4.0f}
    };
    l2Normalize(vec);
    float norm = std::sqrt(std::norm(vec[0]) + std::norm(vec[1]));
    EXPECT_NEAR(norm, 1.0f, 1e-5f);
}

TEST(ComplexDistanceTest, ComputesComplexEuclideanDistance) {
    std::vector<std::complex<float>> a = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };
    std::vector<std::complex<float>> b = {
        {1.0f, 2.0f},
        {4.0f, 6.0f}
    };
    float dist = euclideanDistance(a, b);
    // diff = (0,0), (-1,-2) => norm = 1^2 + 2^2 = 5
    EXPECT_NEAR(dist, std::sqrt(5.0f), 1e-5f);
}

TEST(DominantOrientationTest, FindsMaxBinOrientation) {
    std::vector<float> hist(36, 0.0f);
    hist[9] = 5.0f;  // Dominant bin
    float angle = findDominantOrientation(hist);
    EXPECT_NEAR(angle, 90.0f, 1e-5f);  // 360 / 36 * 9
}

TEST(MatchingTest, MatchesDescriptorSetsCorrectly) {
    Desc d1{{ {0.0f, 0.0f}, {1.0f, 0.0f}, {2.0f, 0.0f} }, 0.0f};
    Desc d2{{ {0.0f, 0.0f}, {1.1f, 0.0f}, {2.1f, 0.0f} }, 0.0f};  // Close to d1
    Desc d3{{ {10.0f, 0.0f}, {10.0f, 0.0f}, {10.0f, 0.0f} }, 0.0f};  // Far from d1

    std::vector<Desc> set1 = {d1};
    std::vector<Desc> set2 = {d2, d3};

    auto matches = matchDescriptorSets(set1, set2, 0.9f);

    ASSERT_EQ(matches.size(), 1);
    EXPECT_EQ(matches[0].idx1, 0);
    EXPECT_EQ(matches[0].idx2, 0);
}

