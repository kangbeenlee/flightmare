#include "flightlib/sensors/stereo_camera.hpp"

namespace flightlib {

StereoCamera::StereoCamera() {}
StereoCamera::~StereoCamera() {}

void StereoCamera::init(const Ref<Vector<3>> B_r_LC, const Ref<Vector<3>> B_r_RC, const Ref<Matrix<3, 3>> R_B_C)
{
    // Camera intrinsic matrix
    K_ = (Matrix<3, 3>() << f_, 0, c_x_, 0, f_, c_y_, 0, 0, 1).finished();

    // Transformation from body to left camera
    T_B_LC_.block<3, 3>(0, 0) = R_B_C;
    T_B_LC_.block<3, 1>(0, 3) = B_r_LC;
    T_B_LC_.row(3) << 0.0, 0.0, 0.0, 1.0;

    // Transformation from left camera to body
    T_LC_B_ = T_B_LC_.inverse();

    // Transformation from body to right camera
    T_B_RC_.block<3, 3>(0, 0) = R_B_C;
    T_B_RC_.block<3, 1>(0, 3) = B_r_RC;
    T_B_RC_.row(3) << 0.0, 0.0, 0.0, 1.0;

    initialized_ = true;
}

bool StereoCamera::computePixelPoint(const Ref<Vector<3>> position, const Ref<Matrix<4, 4>> T_W_B)
{
    if (!initialized_) throw std::runtime_error("Stereo camera is not initialized!");

    // Covert to homogeneous coordinates
    Vector<4> P_W(position[0], position[1], position[2], 1);
    
    // From world to left camera
    Matrix<3, 4> T_W_LC = (T_B_LC_ * T_W_B).topRows<3>();

    // From world to right camera
    Matrix<3, 4> T_W_RC = (T_B_RC_ * T_W_B).topRows<3>();

    // From world cooridnates to camera coordinates (before scaling)
    Vector<3> p_l = K_ * T_W_LC * P_W;
    Vector<3> p_r = K_ * T_W_RC * P_W;

    // If target position is out of field of view, then impossible to detect
    if (p_l[2] < 0 || p_r[2] < 0)
        return false;

    // Scaling pixel coordinates (u, v)
    Scalar u_l = p_l[0] / p_l[2];
    Scalar v_l = p_l[1] / p_l[2];
    Scalar w_l = p_l[2] / p_l[2];
    
    Scalar u_r = p_r[0] / p_r[2];
    Scalar v_r = p_r[1] / p_r[2];
    Scalar w_r = p_r[2] / p_r[2];

    // If target position is out of field of view, then impossible to detect
    if (u_l < 0 || v_l < 0 || w_l < 0 || u_r < 0 || v_r < 0 || w_r < 0 ||
        u_l >= width_ || v_l >= height_ || u_r >= width_ || v_r >= height_)
        return false;

    //
    gt_pixels_ = Vector<4>(u_l, v_l, u_r, v_r);

    // Add gaussian noise to gt pixel
    u_l += norm_dist_(random_gen_);
    v_l += norm_dist_(random_gen_);
    u_r += norm_dist_(random_gen_);
    v_r += norm_dist_(random_gen_);

    //
    pixels_ = Vector<4>(u_l, v_l, u_r, v_r);

    // Compute target coordinates w.r.t. left camera frame
    Scalar x_c = b_ * (u_l - c_x_) / (u_l - u_r);
    Scalar y_c = b_ * f_ * (v_l - c_y_) / (f_ * (u_l - u_r));
    Scalar z_c = b_ * f_ / (u_l - u_r);
    
    Vector<4> P_LC(x_c, y_c, z_c, 1);

    // From left camera to world
    Matrix<3, 4> T_LC_W = (T_B_LC_ * T_W_B).inverse().topRows<3>();
    p_w_ = T_LC_W * P_LC;

    return true;
}

}  // namespace flightlib