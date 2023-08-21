#include "flightlib/sensors/stereo_camera.hpp"

namespace flightlib {

StereoCamera::StereoCamera()
{
    // Transformation from body to left camera
    B_r_LC_ = Vector<3>(0.06, -0.015, 0);
    R_B_LC_ = (Matrix<3, 3>() << 0, -1, 0, 0, 0, -1, 1, 0, 0).finished();
    T_B_LC_.block<3, 3>(0, 0) = R_B_LC_;
    T_B_LC_.block<3, 1>(0, 3) = B_r_LC_;
    T_B_LC_.row(3) << 0.0, 0.0, 0.0, 1.0;

    T_LC_B_ = T_B_LC_.inverse();

    // Transformation from body to right camera
    B_r_RC_ = Vector<3>(-0.06, -0.015, 0);
    R_B_RC_ = (Matrix<3, 3>() << 0, -1, 0, 0, 0, -1, 1, 0, 0).finished();
    T_B_RC_.block<3, 3>(0, 0) = R_B_RC_;
    T_B_RC_.block<3, 1>(0, 3) = B_r_RC_;
    T_B_RC_.row(3) << 0.0, 0.0, 0.0, 1.0;

    // Camera intrinsic matrix
    K_ = (Matrix<3, 3>() << f_, 0, c_, 0, f_, c_, 0, 0, 1).finished();
}


StereoCamera::~StereoCamera() {}

bool StereoCamera::processImagePoint(const Ref<Vector<3>> target_point, const Ref<Matrix<4, 4>> T_W_B)
{
    // Covert to homogeneous coordinates
    Vector<4> P_W(target_point[0], target_point[1], target_point[2], 1);
    
    // From world to left camera
    Matrix<3, 4> T_W_LC = (T_B_LC_ * T_W_B).topRows<3>();

    // From world to right camera
    Matrix<3, 4> T_W_RC = (T_B_RC_ * T_W_B).topRows<3>();

    // From world coordinates to image pixel coordinates (before scaling)
    Vector<3> p_l = K_ * T_W_LC * P_W;
    Vector<3> p_r = K_ * T_W_RC * P_W;

    // Scaling pixel coordinates (u, v)
    Scalar u_l = p_l[0] / p_l[2];
    Scalar v_l = p_l[1] / p_l[2];
    Scalar w_l = p_l[2] / p_l[2];
    
    Scalar u_r = p_r[0] / p_r[2];
    Scalar v_r = p_r[1] / p_r[2];
    Scalar w_r = p_r[2] / p_r[2];

    // if target point is out of field of view, then impossible to detect
    if (u_l < 0 || v_l < 0 || w_l < 0 || u_r < 0 || v_r < 0 || w_r < 0 ||
        u_l >= width_ || v_l >= height_ || u_r >= width_ || v_r >= height_)
        return false;

    //
    gt_pixels_ = Vector<4>(u_l, v_l, u_r, v_r);

    // Compute target coordinates w.r.t. left camera frame
    Scalar x_c = b_ * (u_l - c_) / (u_l - u_r);
    Scalar y_c = b_ * f_ * (v_l - c_) / (f_ * (u_l - u_r));
    Scalar z_c = b_ * f_ / (u_l - u_r);
    gt_p_c_ = Vector<3>(x_c, y_c, z_c);

    // Add noise to gt pixel
    u_l += norm_dist_(random_gen_);
    v_l += norm_dist_(random_gen_);
    u_r += norm_dist_(random_gen_);
    v_r += norm_dist_(random_gen_);

    //
    pixels_ = Vector<4>(u_l, v_l, u_r, v_r);

    // Compute target coordinates w.r.t. left camera frame
    x_c = b_ * (u_l - c_) / (u_l - u_r);
    y_c = b_ * f_ * (v_l - c_) / (f_ * (u_l - u_r));
    z_c = b_ * f_ / (u_l - u_r);
    p_c_ = Vector<3>(x_c, y_c, z_c);

    return true;
}

}  // namespace flightlib