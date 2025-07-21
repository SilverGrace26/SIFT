#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "refine.hpp"

namespace refine {

// -------- Matrix Math Utilities --------
float calculateDeterminant(std::vector<std::vector<float>>& Hessian) {
    float det = 0;
    for (int col = 0; col < 3; col++) {
        float sum = Hessian[0][col] * (
            Hessian[1][(col + 1) % 3] * Hessian[2][(col + 2) % 3] -
            Hessian[1][(col + 2) % 3] * Hessian[2][(col + 1) % 3]);
        det += sum;
    }
    return det;
}

std::vector<std::vector<float>> calculateAdjugate(std::vector<std::vector<float>>& Hessian) {
    std::vector<std::vector<float>> Adjugate(3, std::vector<float>(3));

    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            float cofactor = Hessian[(row + 1) % 3][(col + 1) % 3] * Hessian[(row + 2) % 3][(col + 2) % 3]
                           - Hessian[(row + 1) % 3][(col + 2) % 3] * Hessian[(row + 2) % 3][(col + 1) % 3];
            Adjugate[col][row] = cofactor; 
        }
    }
    return Adjugate;
}

std::vector<std::vector<float>> calculateInverse(std::vector<std::vector<float>>& Hessian) {
    std::vector<std::vector<float>> Adjugate = calculateAdjugate(Hessian);
    float det = calculateDeterminant(Hessian);
    if (std::abs(det) < 1e-6f) {
        return std::vector<std::vector<float>>(3, std::vector<float>(3, 0));
    }

    std::vector<std::vector<float>> Inverse(3, std::vector<float>(3));
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            Inverse[row][col] = Adjugate[row][col] / det;
        }
    }
    return Inverse;
}

// -------- DoG keypoint refinement --------

float accessScaleSpace(const ss::ScaleSpace& DoG_scale_space, const kp::KeyPoint& keypoint) {
    if (keypoint.octave_idx < 0 || keypoint.octave_idx >= DoG_scale_space.size()) {
        throw std::out_of_range("Octave index is out of bounds for ScaleSpace access.");
    }

    const ss::Octave& octave = DoG_scale_space[keypoint.octave_idx];

    if (keypoint.scale_idx < 0 || keypoint.scale_idx >= octave.size()) {
        throw std::out_of_range("Scale index is out of bounds for Octave access.");
    }

    const auto& img = octave[static_cast<int>(std::round(keypoint.scale_idx))]; 

    // Checking X and Y bounds for the actual access
    int access_x = static_cast<int>(std::round(keypoint.x));
    int access_y = static_cast<int>(std::round(keypoint.y));

    if (access_y < 0 || access_y >= img.rows ||
        access_x < 0 || access_x >= img.cols) {
        throw std::out_of_range("Keypoint (x,y) coordinates are out of bounds for image access."); 
    }
    return img.at<float>(access_y, access_x);
}

auto make_dog_accessor = [](const ss::ScaleSpace& ss, const kp::KeyPoint& kp) {
    return [&ss, kp](int dx, int dy, int ds) -> float {
        kp::KeyPoint neighbor_kp = kp;
        neighbor_kp.x += dx;
        neighbor_kp.y += dy;
        neighbor_kp.scale_idx += ds;

        // getting specific octave to check scale index bounds
        if (neighbor_kp.octave_idx < 0 || neighbor_kp.octave_idx >= ss.size()) {
            return 0.0f; 
        }

        const ss::Octave& current_octave = ss[neighbor_kp.octave_idx];

        // if the neighbor's scale_idx is valid for the current octave
        if (neighbor_kp.scale_idx < 0.0f || neighbor_kp.scale_idx >= current_octave.size()) {
            return 0.0f; 
        }

        const cv::Mat& target_img = current_octave[static_cast<int>(std::round(neighbor_kp.scale_idx))];
        int neighbor_access_x = static_cast<int>(std::round(neighbor_kp.x));
        int neighbor_access_y = static_cast<int>(std::round(neighbor_kp.y));

        if (neighbor_access_y < 0 || neighbor_access_y >= target_img.rows ||
            neighbor_access_x < 0 || neighbor_access_x >= target_img.cols) {
            return 0.0f;
        }

        return target_img.at<float>(neighbor_access_y, neighbor_access_x);
    };
};

std::vector<float> calculateKeypointGradients(const ss::ScaleSpace& DoG_scale_space, const kp::KeyPoint& keypoint){
    auto dog = make_dog_accessor(DoG_scale_space, keypoint);
    
    std::vector<float> gradients(3);

    gradients[0] = 0.5f * (dog(1, 0, 0) - dog(-1, 0, 0));
    gradients[1] = 0.5f * (dog(0, 1, 0) - dog(0, -1, 0));
    gradients[2] = 0.5f * (dog(0, 0, 1) - dog(0, 0, -1));

    return gradients;
};

std::vector<std::vector<float>> calculateKeypointHessian (const ss::ScaleSpace& DoG_scale_space, const kp::KeyPoint& keypoint) {
    auto dog = make_dog_accessor(DoG_scale_space, keypoint);

    std::vector<float> curvature(3);


    curvature[0] = dog(1, 0, 0) + dog(-1, 0, 0) - 2 * dog(0, 0, 0);
    curvature[1] = dog(0, 1, 0) + dog(0, -1, 0) - 2 * dog(0, 0, 0);
    curvature[2] = dog(0, 0, 1) + dog(0, 0, -1) - 2 * dog(0, 0, 0);

    float dxy = 0.25f * (dog(1, 1, 0) - dog(1, -1, 0) - dog(-1, 1, 0) + dog(-1, -1, 0));
    float dxs = 0.25f * (dog(1, 0, 1) - dog(1, 0, -1) - dog(-1, 0, 1) + dog(-1, 0, -1));
    float dys = 0.25f * (dog(0, 1, 1) - dog(0, 1, -1) - dog(0, -1, 1) + dog(0, -1, -1));

    std::vector<std::vector<float>> hessian = {
        {curvature[0], dxy, dxs},
        {dxy, curvature[1], dys},
        {dxs, dys, curvature[2]}
    };

    return hessian;
};

bool isOnEdge(const std::vector<std::vector<float>>& hessian, float edge_threshold) {

    float dxx = hessian[0][0];
    float dyy = hessian[1][1];
    float dxy = hessian[0][1];
    
    float det = dxx * dyy - dxy * dxy;
    float trace = dxx + dyy;
    
    if (std::abs(det) < 1e-6) {
        return false;  
    }
    
    float r = edge_threshold;
    float criterion = (trace * trace) / det;
    
    return criterion < (r + 1) * (r + 1) / r;  
};

void refineKeypoints(const ss::ScaleSpace& DoG_scale_space, kp::KeyPoint& keypoint) {

    const auto& current_octave_DoG = DoG_scale_space[keypoint.octave_idx];

    if (keypoint.scale_idx <= 0 || keypoint.scale_idx >= (current_octave_DoG.size() - 1)) {
        // This keypoint is at a scale boundary, cannot compute full 3x3x3 derivatives.
        keypoint.x = -1e6; 
        return;
    }

    std::vector<float> gradients = calculateKeypointGradients(DoG_scale_space, keypoint);
    std::vector<std::vector<float>> hessian = calculateKeypointHessian(DoG_scale_space, keypoint);

    float det_hessian = calculateDeterminant(hessian);
    if (std::abs(det_hessian) < 1e-6f) { 
        keypoint.x = -1e6; 
        return;
    }

    std::vector<std::vector<float>> Inverse = calculateInverse(hessian);


    std::vector<float> offset(3, 0.0f);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            offset[i] -= Inverse[i][j] * gradients[j];
        }
    }

    // Apply offset if it's within limits
    if (std::abs(offset[0]) < 0.5f && std::abs(offset[1]) < 0.5f && std::abs(offset[2]) < 0.5f) {
        keypoint.x += offset[0];
        keypoint.y += offset[1];
        keypoint.scale_idx += offset[2];

        // We have to recheck after refining
        int rounded_octave_idx = static_cast<int>(keypoint.octave_idx); 
        int rounded_scale_idx = static_cast<int>(std::round(keypoint.scale_idx));

        // Get the dimensions of the image this refined keypoint "belongs" to
        if (rounded_octave_idx < 0 || rounded_octave_idx >= DoG_scale_space.size() ||
            rounded_scale_idx < 0 || rounded_scale_idx >= DoG_scale_space[rounded_octave_idx].size()) {
            keypoint.x = -1e6; 
            return;
        }

        const cv::Mat& relevant_img = DoG_scale_space[rounded_octave_idx][rounded_scale_idx];

        if (keypoint.x < 0 || keypoint.x >= relevant_img.cols - 1 || 
            keypoint.y < 0 || keypoint.y >= relevant_img.rows - 1 ||
            keypoint.scale_idx < 0.0f || keypoint.scale_idx >= (DoG_scale_space[rounded_octave_idx].size() - 1)) 
        {
             keypoint.x = -1e6; 
             return;
        }

        // Low contrast check 
        float refined_dog_val = accessScaleSpace(DoG_scale_space, keypoint);
        if (std::abs(refined_dog_val) < 0.03f) { 
             keypoint.x = -1e6; 
             return;
        }

    } else {
        // Offset is too large, keypoint not well localized 
        keypoint.x = -1e6;
        keypoint.y = -1e6;
        return;
    }

    if (isOnEdge(hessian, 10.0f)) { // Typically r = 10, so (r+1)^2/r = 121/10 = 12.1
        keypoint.x = -1e6; 
        return;
    }
}

}



