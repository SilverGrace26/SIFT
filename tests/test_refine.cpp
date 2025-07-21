#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <glm/glm.hpp>
#include "refine.hpp"

std::vector<std::vector<float>> createTestMatrix3x3() {
    return {
        {1.3f, 4.77f, 5.89f},
        {5.73f, 6.32f, -7.23f},
        {-1.342f, 8.88f, -9.0f}
    };
}

void ExpectMatricesNear(const std::vector<std::vector<float>>& A,
                        const std::vector<std::vector<float>>& B,
                        float tolerance) {
    ASSERT_EQ(A.size(), B.size());
    for (size_t i = 0; i < A.size(); ++i) {
        ASSERT_EQ(A[i].size(), B[i].size());
        for (size_t j = 0; j < A[i].size(); ++j) {
            EXPECT_NEAR(A[i][j], B[i][j], tolerance);
        }
    }
}

TEST(MatUtilsTest, ReturnsDeterminantValue) {
    std::vector<std::vector<float>> test = createTestMatrix3x3();
    glm::mat3 glm_mat(
        test[0][0], test[0][1], test[0][2],
        test[1][0], test[1][1], test[1][2],
        test[2][0], test[2][1], test[2][2]
    );

    float expected = glm::determinant(glm_mat);
    float result = refine::calculateDeterminant(test);

    EXPECT_NEAR(result, expected, 1e-3f);
}

TEST(MatUtilsTest, ReturnsInverseValue){
    std::vector<std::vector<float>> test = createTestMatrix3x3();
    auto inv = refine::calculateInverse(test);

    glm::mat3 glm_mat(
        test[0][0], test[0][1], test[0][2],
        test[1][0], test[1][1], test[1][2],
        test[2][0], test[2][1], test[2][2]
    );

    glm::mat3 glm_inv = glm::inverse(glm_mat);

    std::vector<std::vector<float>> expected(3, std::vector<float>(3));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            expected[i][j] = glm_inv[i][j];

    ExpectMatricesNear(inv, expected, 1e-2f);
}

// Adjugate comparison using glm: verify inv * det == adjugate
TEST(MatUtilsTest, ReturnsAdjugateValue){
    std::vector<std::vector<float>> test = createTestMatrix3x3();
    auto adj = refine::calculateAdjugate(test);

    glm::mat3 glm_mat(
        test[0][0], test[0][1], test[0][2],
        test[1][0], test[1][1], test[1][2],
        test[2][0], test[2][1], test[2][2]
    );

    float det = glm::determinant(glm_mat);
    glm::mat3 glm_adj = glm::inverse(glm_mat) * det;

    std::vector<std::vector<float>> expected(3, std::vector<float>(3));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            expected[i][j] = glm_adj[i][j];

    ExpectMatricesNear(adj, expected, 1e-1f);  // Adjugate is less numerically stable
}

// --- Remaining tests unchanged ---

TEST(KeypointRefinementTest, AccessScaleSpaceReturnsCorrectValue) {
    ss::ScaleSpace ss(1, std::vector<cv::Mat>(1, cv::Mat::zeros(5, 5, CV_32F)));
    ss[0][0].at<float>(2, 3) = 42.0f;

    kp::KeyPoint kp = {3, 2, 0, 0};  // y=2, x=3

    float val = refine::accessScaleSpace(ss, kp);
    EXPECT_FLOAT_EQ(val, 42.0f);
}

ss::ScaleSpace createTestScaleSpace() {
    ss::ScaleSpace ss(1); // 1 octave
    ss[0].resize(3); // 3 scale levels

    for (int i = 0; i < 3; ++i) {
        ss[0][i] = cv::Mat::zeros(5, 5, CV_32F);
    }

    // Create a peak at (2, 2) in the middle scale
    ss[0][1].at<float>(2, 2) = 10.0f;
    ss[0][1].at<float>(1, 2) = 8.0f;
    ss[0][1].at<float>(3, 2) = 8.0f;
    ss[0][1].at<float>(2, 1) = 8.0f;
    ss[0][1].at<float>(2, 3) = 8.0f;

    // Simulate contrast in adjacent scales
    ss[0][0].at<float>(2, 2) = 5.0f;
    ss[0][2].at<float>(2, 2) = 6.0f;

    return ss;
}

TEST(KeypointRefinementTest, CalculatesCorrectGradients) {
    auto ss = createTestScaleSpace();
    kp::KeyPoint kp{2, 2, 1, 0};  // x, y, scale_idx, octave_idx

    auto gradients = refine::calculateKeypointGradients(ss, kp);

    EXPECT_NEAR(gradients[0], 0.0f, 1e-5f);
    EXPECT_NEAR(gradients[1], 0.0f, 1e-5f);
    EXPECT_NEAR(gradients[2], 0.5f * (6.0f - 5.0f), 1e-5f); // 0.5
}
