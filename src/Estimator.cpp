/**
 * @file      Estimator.cpp
 * @brief     Implementation of tightly-coupled LiDAR-Inertial Odometry Estimator
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright  Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "Estimator.h"
#include "LieUtils.h"
#include "PointCloudUtils.h"
#include <spdlog/spdlog.h>
#include <chrono>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace lio {

// ============================================================================
// HARDCODED EXTRINSICS FOR R3LIVE DATASET (Livox Avia sensor)
// Frame convention: T_il = Transform from LiDAR to IMU
// ============================================================================
namespace Extrinsics {
    // Translation from LiDAR to IMU (in meters)
    const Eigen::Vector3f t_il(0.04165f, 0.02326f, -0.0284f);
    
    // Rotation from LiDAR to IMU (identity - sensors are aligned)
    const Eigen::Matrix3f R_il = Eigen::Matrix3f::Identity();
}

// ============================================================================
// Constructor & Destructor
// ============================================================================

Estimator::Estimator()
    : m_current_state()
    , m_initialized(false)
    , m_last_update_time(0.0)
    , m_frame_count(0)
    , m_first_lidar_frame(true)
    , m_last_lidar_time(0.0)
    , m_first_keyframe(true)
    , m_last_keyframe_position(Eigen::Vector3f::Zero())
    , m_last_keyframe_rotation(Eigen::Matrix3f::Identity())
{
    // Initialize process noise matrix (Q)
    m_process_noise = Eigen::Matrix<float, 18, 18>::Identity();
    m_process_noise.block<3,3>(0,0) *= m_params.gyr_noise_std * m_params.gyr_noise_std;
    m_process_noise.block<3,3>(3,3) *= m_params.acc_noise_std * m_params.acc_noise_std;
    m_process_noise.block<3,3>(6,6) *= m_params.acc_noise_std * m_params.acc_noise_std;
    m_process_noise.block<3,3>(9,9) *= m_params.gyr_bias_noise_std * m_params.gyr_bias_noise_std;
    m_process_noise.block<3,3>(12,12) *= m_params.acc_bias_noise_std * m_params.acc_bias_noise_std;
    m_process_noise.block<3,3>(15,15) *= m_params.gravity_noise_std * m_params.gravity_noise_std;
    
    // Initialize state transition matrix
    m_state_transition = Eigen::Matrix<float, 18, 18>::Identity();
    
    // Initialize local map
    m_map_cloud = std::make_shared<PointCloud>();
    
    // Initialize statistics
    m_statistics = Statistics();
    m_statistics.total_frames = 0;
    m_statistics.successful_registrations = 0;
    m_statistics.avg_processing_time_ms = 0.0;
    m_statistics.total_distance = 0.0;
    m_statistics.avg_translation_error = 0.0;
    m_statistics.avg_rotation_error = 0.0;
    
    spdlog::info("[Estimator] Initialized with hardcoded extrinsics (R3LIVE/Avia dataset)");
    spdlog::info("[Estimator] t_il = [{:.5f}, {:.5f}, {:.5f}]", 
                 Extrinsics::t_il.x(), Extrinsics::t_il.y(), Extrinsics::t_il.z());
    spdlog::info("[Estimator] R_il = Identity");
}

Estimator::~Estimator() {
    std::lock_guard<std::mutex> lock_state(m_state_mutex);
    std::lock_guard<std::mutex> lock_map(m_map_mutex);
    std::lock_guard<std::mutex> lock_stats(m_stats_mutex);
}

// ============================================================================
// Initialization
// ============================================================================

bool Estimator::GravityInitialization(const std::vector<IMUData>& imu_buffer) {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    
    if (m_initialized) {
        spdlog::warn("[Estimator] Already initialized!");
        return false;
    }
    
    // 1. Check minimum number of samples (need at least 20 for good statistics)
    if (imu_buffer.size() < 20) {
        spdlog::error("[Estimator] Not enough IMU data for initialization (need >= 20 samples, got {})", 
                     imu_buffer.size());
        return false;
    }
    
    spdlog::info("[Estimator] Starting gravity initialization with {} IMU samples", imu_buffer.size());
    
    // 2. Compute mean acceleration and gyroscope (running average)
    Eigen::Vector3f mean_acc = Eigen::Vector3f::Zero();
    Eigen::Vector3f mean_gyr = Eigen::Vector3f::Zero();
    
    for (const auto& imu : imu_buffer) {
        mean_acc += imu.acc;
        mean_gyr += imu.gyr;
    }
    mean_acc /= static_cast<float>(imu_buffer.size());
    mean_gyr /= static_cast<float>(imu_buffer.size());
    
    // 3. Compute variance to check if robot is stationary
    float acc_variance = 0.0f;
    float gyr_variance = 0.0f;
    
    for (const auto& imu : imu_buffer) {
        acc_variance += (imu.acc - mean_acc).squaredNorm();
        gyr_variance += (imu.gyr - mean_gyr).squaredNorm();
    }
    acc_variance /= static_cast<float>(imu_buffer.size());
    gyr_variance /= static_cast<float>(imu_buffer.size());
    
    // 4. Check if robot is stationary (low variance)
    if (acc_variance > 0.5f) {
        spdlog::warn("[Estimator] High accelerometer variance ({:.3f}), robot may be moving!", acc_variance);
        spdlog::warn("[Estimator] Initialization may be inaccurate. Please keep robot stationary.");
    }
    
    if (gyr_variance > 0.01f) {
        spdlog::warn("[Estimator] High gyroscope variance ({:.3f}), robot may be rotating!", gyr_variance);
    }
    
    // 5. Initialize state
    m_current_state.Reset();
    
    // 6. Check accelerometer norm (should be ~9.81 m/s² if stationary)
    float acc_norm = mean_acc.norm();
    
    if (std::abs(acc_norm - 9.81f) > 1.5f) {
        spdlog::error("[Estimator] Accelerometer norm = {:.3f} m/s² (expected ~9.81)", acc_norm);
        spdlog::error("[Estimator] Sensor may be moving or miscalibrated. Initialization failed.");
        return false;
    }
    
    // 7. Initialize gravity vector (measured acceleration = -gravity in sensor frame)
    Eigen::Vector3f gravity_measured = -mean_acc.normalized() * 9.81f;
    
    // 8. Set initial gravity (not yet aligned)
    m_current_state.m_gravity = gravity_measured;
    
    // 9. Initialize rotation to identity (will be aligned after)
    m_current_state.m_rotation = Eigen::Matrix3f::Identity();
    
    spdlog::info("[Estimator] Initial gravity (sensor frame): [{:.3f}, {:.3f}, {:.3f}]", 
                 gravity_measured.x(), gravity_measured.y(), gravity_measured.z());
    
    // 10. Gravity alignment: align world frame so gravity points to [0, 0, -9.81]
    // This rotates all states to make gravity vertical
    Eigen::Vector3f gravity_target(0.0f, 0.0f, -9.81f);
    Eigen::Quaternionf q_align = Eigen::Quaternionf::FromTwoVectors(
        m_current_state.m_gravity.normalized(),
        gravity_target.normalized()
    );
    Eigen::Matrix3f R_align = q_align.toRotationMatrix();
    
   
    
    // Apply alignment rotation to all states
    m_current_state.m_rotation = R_align * m_current_state.m_rotation;  // Rotate orientation
    m_current_state.m_position = R_align * m_current_state.m_position;  // Rotate position (zero)
    m_current_state.m_velocity = R_align * m_current_state.m_velocity;  // Rotate velocity (zero)
    m_current_state.m_gravity = R_align * m_current_state.m_gravity;    // Rotate gravity -> [0,0,-9.81]
    
  
    // 11. Initialize gyroscope bias (stationary gyro reading = bias)
    m_current_state.m_gyro_bias = mean_gyr;
    
    // 12. Initialize accelerometer bias from stationary measurements
    // Stationary condition: acc_measured = -g + bias
    // After gravity alignment: mean_acc ≈ -R_align^T * g_world + bias
    // Therefore: bias = mean_acc + R_align^T * g_world
    //                 = mean_acc + R_align^T * [0, 0, -9.81]
    Eigen::Vector3f g_aligned(0.0f, 0.0f, -9.81f);

    // Correct formula: bias = mean_acc + R^T * g
    Eigen::Vector3f acc_bias_estimate = mean_acc + m_current_state.m_rotation.transpose() * g_aligned;
    m_current_state.m_acc_bias = acc_bias_estimate;
    
    // 13. Initialize position and velocity to zero
    m_current_state.m_position.setZero();
    m_current_state.m_velocity.setZero();
    
    // 14. Initialize covariance with appropriate uncertainty
    m_current_state.m_covariance = Eigen::Matrix<float, 18, 18>::Identity();
    m_current_state.m_covariance.block<3,3>(0,0) *= 0.01f;   // rotation (small, well aligned)
    m_current_state.m_covariance.block<3,3>(3,3) *= 1.0f;    // position (unknown)
    m_current_state.m_covariance.block<3,3>(6,6) *= 0.1f;    // velocity (should be zero)
    m_current_state.m_covariance.block<3,3>(9,9) *= 0.001f;  // gyro bias (estimated from data)
    m_current_state.m_covariance.block<3,3>(12,12) *= 0.01f; // acc bias (estimated from data)
    m_current_state.m_covariance.block<3,3>(15,15) *= 0.001f; // gravity (well aligned)
    
    // 15. Set timestamp
    m_last_update_time = imu_buffer.back().timestamp;
    
    // 16. Mark as initialized
    m_initialized = true;
    
    spdlog::info("[Estimator] ═══════════════════════════════════════════════════════");
    spdlog::info("[Estimator] Gravity initialization SUCCESSFUL at t={:.6f}", m_last_update_time);
    spdlog::info("[Estimator] Statistics:");
    spdlog::info("  - IMU samples: {}", imu_buffer.size());
    spdlog::info("  - Acc variance: {:.6f} m²/s⁴", acc_variance);
    spdlog::info("  - Gyr variance: {:.6f} rad²/s²", gyr_variance);
    spdlog::info("  - Acc norm: {:.3f} m/s² (expected: 9.81)", acc_norm);
    spdlog::info("[Estimator] ═══════════════════════════════════════════════════════");
    
    return true;
}

void Estimator::Initialize(const IMUData& first_imu) {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    
    if (m_initialized) {
        spdlog::warn("[Estimator] Already initialized!");
        return;
    }
    
    spdlog::warn("[Estimator] Simple initialization with single IMU sample");
    spdlog::warn("[Estimator] Consider using GravityInitialization() with multiple samples for better accuracy");
    
    // Initialize state with first IMU measurement
    m_current_state.Reset();
    
    // Initial gravity alignment (assume stationary)
    Eigen::Vector3f acc_world = first_imu.acc;
    float acc_norm = acc_world.norm();
    
    if (std::abs(acc_norm - 9.81f) < 1.0f) {
        // Use accelerometer to initialize gravity direction
        m_current_state.m_gravity = -acc_world.normalized() * 9.81f;
        
        // Gravity alignment: rotate world frame so gravity points to [0, 0, -9.81]
        Eigen::Vector3f gravity_target(0.0f, 0.0f, -9.81f);
        Eigen::Quaternionf q_align = Eigen::Quaternionf::FromTwoVectors(
            m_current_state.m_gravity.normalized(),
            gravity_target.normalized()
        );
        Eigen::Matrix3f R_align = q_align.toRotationMatrix();
        
        // Apply alignment to initial rotation
        m_current_state.m_rotation = R_align;
        m_current_state.m_gravity = gravity_target;
        
        spdlog::info("[Estimator] Gravity initialized: [{:.3f}, {:.3f}, {:.3f}]",
                     m_current_state.m_gravity.x(), 
                     m_current_state.m_gravity.y(), 
                     m_current_state.m_gravity.z());
    } else {
        spdlog::warn("[Estimator] Accelerometer norm = {:.3f} (expected ~9.81). Using default gravity.", acc_norm);
        m_current_state.m_gravity = Eigen::Vector3f(0.0f, 0.0f, -9.81f);
        m_current_state.m_rotation = Eigen::Matrix3f::Identity();
    }
    
    // Initialize biases to zero (will be estimated)
    m_current_state.m_gyro_bias.setZero();
    m_current_state.m_acc_bias.setZero();
    
    // Initialize position and velocity
    m_current_state.m_position.setZero();
    m_current_state.m_velocity.setZero();
    
    // Initialize covariance with large uncertainty
    m_current_state.m_covariance = Eigen::Matrix<float, 18, 18>::Identity();
    m_current_state.m_covariance.block<3,3>(0,0) *= 0.1f;    // rotation
    m_current_state.m_covariance.block<3,3>(3,3) *= 1.0f;    // position
    m_current_state.m_covariance.block<3,3>(6,6) *= 0.5f;    // velocity
    m_current_state.m_covariance.block<3,3>(9,9) *= 0.01f;   // gyro bias
    m_current_state.m_covariance.block<3,3>(12,12) *= 0.1f;  // acc bias
    m_current_state.m_covariance.block<3,3>(15,15) *= 0.01f; // gravity
    
    m_last_update_time = first_imu.timestamp;
    
    m_initialized = true;
    spdlog::info("[Estimator] Initialization complete at t={:.6f}", first_imu.timestamp);
}

// ============================================================================
// IMU Processing (Forward Propagation)
// ============================================================================

void Estimator::ProcessIMU(const IMUData& imu_data) {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    
    if (!m_initialized) {
        spdlog::warn("[Estimator] Not initialized. Call Initialize() first.");
        return;
    }
    
    // Propagate state using current IMU measurement
    PropagateState(imu_data);
    
    // Save state history for undistortion (keep last 0.5 seconds)
    // This is more efficient than keeping raw IMU data
    StateWithTimestamp state_snapshot;
    state_snapshot.state = m_current_state;
    state_snapshot.timestamp = imu_data.timestamp;
    m_state_history.push_back(state_snapshot);
    
    // Clean old states (keep only recent 0.5 seconds for undistortion)
    while (!m_state_history.empty() && 
           imu_data.timestamp - m_state_history.front().timestamp > 0.5) {
        m_state_history.pop_front();
    }
}

void Estimator::PropagateState(const IMUData& imu) {
    // Time step (only timestamp is double)
    double dt = imu.timestamp - m_last_update_time;
    if (dt <= 0.0 || dt > 1.0) {
        spdlog::error("[Estimator] Invalid dt: {:.6f}", dt);
        return;
    }
    float dt_f = static_cast<float>(dt);
    
    // Get current state (all float)
    Eigen::Matrix3f R = m_current_state.m_rotation;
    Eigen::Vector3f p = m_current_state.m_position;
    Eigen::Vector3f v = m_current_state.m_velocity;
    Eigen::Vector3f bg = m_current_state.m_gyro_bias;
    Eigen::Vector3f ba = m_current_state.m_acc_bias;
    Eigen::Vector3f g = m_current_state.m_gravity;
    
    // Corrected measurements (already float from IMUData)
    Eigen::Vector3f omega = imu.gyr - bg;  // angular velocity
    Eigen::Vector3f acc = imu.acc - ba;    // linear acceleration

    Eigen::Vector3f acc_world = R * acc + g;
    
    // --- Forward Propagation (Euler integration) ---
    // dR/dt = R * [omega]_x  =>  R(t+dt) = R(t) * Exp(omega * dt)
    Eigen::Vector3f omega_dt = omega * dt_f;
    Eigen::Matrix3f R_delta = SO3::Exp(omega_dt).Matrix();
    Eigen::Matrix3f R_new = R * R_delta;
    
    // dv/dt = R * acc + g  =>  v(t+dt) = v(t) + (R * acc + g) * dt
    Eigen::Vector3f v_new = v + (R * acc + g) * dt_f;
    
    // dp/dt = v  =>  p(t+dt) = p(t) + v * dt + 0.5 * (R * acc + g) * dt²
    Eigen::Vector3f p_new = p + v * dt_f + 0.5f * (R * acc + g) * dt_f * dt_f;
    
    // Biases: random walk (no change in mean)
    Eigen::Vector3f bg_new = bg;
    Eigen::Vector3f ba_new = ba;
    Eigen::Vector3f g_new = g;
    
    // --- Covariance Propagation ---
    // P(t+dt) = F * P(t) * F^T + Q * dt
    UpdateProcessNoise(dt);
    
    // Build state transition matrix F (18x18)
    // Simplified linearization around current state
    m_state_transition.setIdentity();
    
    // dR depends on omega (rotation dynamics)
    Eigen::Matrix3f omega_skew = Hat(omega);
    m_state_transition.block<3,3>(0,0) = Eigen::Matrix3f::Identity() - omega_skew * dt_f;
    m_state_transition.block<3,3>(0,9) = -R * dt_f;  // rotation vs gyro bias
    
    // dv depends on R and acc (velocity dynamics)
    Eigen::Matrix3f acc_skew = Hat(acc);
    m_state_transition.block<3,3>(6,0) = -R * acc_skew * dt_f;  // velocity vs rotation
    m_state_transition.block<3,3>(6,6) = Eigen::Matrix3f::Identity();
    m_state_transition.block<3,3>(6,12) = -R * dt_f;  // velocity vs acc bias
    m_state_transition.block<3,3>(6,15) = Eigen::Matrix3f::Identity() * dt_f;  // velocity vs gravity
    
    // dp depends on v (position dynamics)
    m_state_transition.block<3,3>(3,3) = Eigen::Matrix3f::Identity();
    m_state_transition.block<3,3>(3,6) = Eigen::Matrix3f::Identity() * dt_f;  // position vs velocity
    
    // Propagate covariance
    Eigen::Matrix<float, 18, 18> P = m_current_state.m_covariance;
    m_current_state.m_covariance = m_state_transition * P * m_state_transition.transpose() 
                                   + m_process_noise * dt_f;
    
    // Update state
    m_current_state.m_rotation = R_new;
    m_current_state.m_position = p_new;
    m_current_state.m_velocity = v_new;
    m_current_state.m_gyro_bias = bg_new;
    m_current_state.m_acc_bias = ba_new;
    m_current_state.m_gravity = g_new;
    
    m_last_update_time = imu.timestamp;
}

// ============================================================================
// LiDAR Processing (Iterated Kalman Update)
// ============================================================================

void Estimator::ProcessLidar(const LidarData& lidar) {
    if (!m_initialized) {
        spdlog::error("[Estimator] Not initialized! Cannot process LiDAR.");
        return;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::lock_guard<std::mutex> lock_state(m_state_mutex);
    std::lock_guard<std::mutex> lock_map(m_map_mutex);
    
    // Downsample input scan for efficiency (do this at the top)
    PointCloudPtr undistorted_cloud = lidar.cloud;
    if (m_params.enable_undistortion) {
        // TODO: Implement motion undistortion using IMU buffer
        // For now, skip undistortion
    }
    
    auto downsampled_scan = std::make_shared<PointCloud>();
    VoxelGrid scan_filter;
    scan_filter.SetInputCloud(undistorted_cloud);
    scan_filter.SetLeafSize(0.4f);  // 20cm voxel size for input scan
    scan_filter.Filter(*downsampled_scan);


// or (size_t i = 0; i < m_cloud->size(); ++i) {
//         float dist_squared = query_point.squared_distance_to(m_cloud->at(i));
//         if (dist_squared <= radius_squared) {

    // range filter to remove far points
    PointCloudPtr range_filtered_scan = std::make_shared<PointCloud>();
    unsigned int initial_size = downsampled_scan->size();
    unsigned int final_size = 0;
    for(unsigned int i = 0; i < initial_size; ++i) {
        const auto& point = downsampled_scan->at(i);
        float range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        if (range <= 50.0f) {
            range_filtered_scan->push_back(point);
            final_size++;
        }
    }

    // for (const auto& point : downsampled_scan->points) {
    //     float range = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
    //     if (range <= 50.0f) {
    //         range_filtered_scan->push_back(point);
    //     }
    // }
    
    
    // spdlog::info("[Estimator] Input scan downsampled: {} -> {} points", 
    //              undistorted_cloud->size(), downsampled_scan->size());
    
    // Create LidarData with downsampled cloud for all processing
    LidarData downsampled_lidar(lidar.timestamp, range_filtered_scan);
    
    // First frame: initialize map with downsampled cloud
    if (m_first_lidar_frame) {
        spdlog::info("[Estimator] First LiDAR frame - initializing map");
        UpdateLocalMap(range_filtered_scan);
        m_first_lidar_frame = false;
        m_last_lidar_time = lidar.timestamp;
        m_last_lidar_state = m_current_state;
        m_frame_count++;
        return;
    }
    
    // Motion check: skip if not enough motion
    float distance = (m_current_state.m_position - m_last_lidar_state.m_position).norm();
  
    
    // Iterated Kalman Filter Update with downsampled data
    UpdateWithLidar(downsampled_lidar);
    
    // Update local map with downsampled scan
    UpdateLocalMap(range_filtered_scan);
    
    // Update statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    double processing_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    {
        std::lock_guard<std::mutex> lock_stats(m_stats_mutex);
        m_processing_times.push_back(processing_time);
        m_statistics.total_frames++;
        m_statistics.avg_processing_time_ms = 
            (m_statistics.avg_processing_time_ms * (m_statistics.total_frames - 1) + processing_time) 
            / m_statistics.total_frames;
    }
    
    // Store trajectory
    m_trajectory.push_back(m_current_state);
    if (m_trajectory.size() > 10000) {
        m_trajectory.pop_front();
    }
    
    // Update tracking
    m_last_lidar_time = lidar.timestamp;
    m_last_lidar_state = m_current_state;
    m_frame_count++;
    
    spdlog::info("[Estimator] Frame {} processed in {:.2f} ms", m_frame_count, processing_time);
}

void Estimator::UpdateWithLidar(const LidarData& lidar) {
    auto start_total = std::chrono::high_resolution_clock::now();
    
    // Debug: Check map status first
    spdlog::info("[UpdateWithLidar] Map status: {} points, KdTree: {}", 
                 m_map_cloud ? m_map_cloud->size() : 0,
                 m_map_kdtree ? "exists" : "null");
    
    // Step 1: Find correspondences (point-to-plane matching)
    auto start_corr = std::chrono::high_resolution_clock::now();
    auto correspondences = FindCorrespondences(lidar.cloud);
    auto end_corr = std::chrono::high_resolution_clock::now();
    double corr_time = std::chrono::duration<double, std::milli>(end_corr - start_corr).count();
    
    if (correspondences.empty()) {
        spdlog::warn("[Estimator] No correspondences found, skipping LiDAR update");
        return;
    }
    
    spdlog::info("[Estimator] Found {} correspondences", correspondences.size());
    spdlog::warn("[UpdateWithLidar] ⏱️  Correspondence finding took {:.2f} ms", corr_time);
    
    // Step 2: Iterative Kalman Filter Update
    auto start_ikf = std::chrono::high_resolution_clock::now();
    const int max_iterations = 10;
    bool converged = false;
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // Compute Jacobian H and residual vector
        Eigen::MatrixXf H;
        Eigen::VectorXf residual;
        ComputeLidarJacobians(correspondences, H, residual);
        
        int num_corr = correspondences.size();
        
        // Compute R_inv for each correspondence individually
        // R_inv(i) = 1.0 / (0.001 + sigma_l + n^T * var * n)
        // For simplicity, use constant noise model
        Eigen::VectorXf R_inv(num_corr);
        for (int i = 0; i < num_corr; i++) {
            // Simple noise model: R_inv(i) = 1.0 / lidar_noise^2
            float sigma = m_params.lidar_noise_std * m_params.lidar_noise_std;
            R_inv(i) = 1.0f / (0.001f + sigma);
        }
        
        // Compute H^T * R_inv (6 x num_corr) - only for rotation and position
        Eigen::MatrixXf H_6 = H.block(0, 0, num_corr, 6);  // Only first 6 columns
        Eigen::MatrixXf H_T_R_inv(6, num_corr);
        for (int i = 0; i < num_corr; i++) {
            H_T_R_inv.col(i) = H_6.row(i).transpose() * R_inv(i);
        }
        
        // Compute H^T * R^-1 * H (6x6)
        Eigen::Matrix<float, 6, 6> H_T_R_inv_H = H_T_R_inv * H_6;
        
        // Compute H^T * R^-1 * z
        Eigen::Matrix<float, 6, 1> H_T_R_inv_z = H_T_R_inv * residual;
        
        // Get prior covariance P (full 18x18)
        Eigen::Matrix<float, 18, 18> P_prior = m_current_state.m_covariance;
        
        // Build H^T*R^-1*H for full state (18x18), only first 6x6 block is non-zero
        Eigen::Matrix<float, 18, 18> H_T_R_inv_H_full = Eigen::Matrix<float, 18, 18>::Zero();
        H_T_R_inv_H_full.block<6, 6>(0, 0) = H_T_R_inv_H;
        
        // Compute Kalman gain using information form:
        // K_1 = (H^T * R^-1 * H + P^-1)^-1  (18x18)
        Eigen::Matrix<float, 18, 18> information_matrix = H_T_R_inv_H_full + P_prior.inverse();
        Eigen::Matrix<float, 18, 18> K_1 = information_matrix.inverse();
        
        // Compute G matrix (18x18): G = K_1 * H^T*R^-1*H
        Eigen::Matrix<float, 18, 18> G = Eigen::Matrix<float, 18, 18>::Zero();
        G.block<18, 6>(0, 0) = K_1.block<18, 6>(0, 0) * H_T_R_inv_H;
        
        // Compute state correction (18-dim) with covariance coupling
        // solution = K_1[18x6] * HTz[6x1] (only rot/pos directly observed, but biases updated via coupling)
        Eigen::Matrix<float, 18, 1> dx = K_1.block<18, 6>(0, 0) * H_T_R_inv_z;
        
        // Apply state correction
        ApplyStateCorrection(dx);
        
        // Check convergence
        float rot_change = dx.segment<3>(0).norm() * 57.3f;  // degrees
        float pos_change = dx.segment<3>(3).norm() * 100.0f;  // cm
        float vel_change = dx.segment<3>(6).norm() * 100.0f;  // cm/s
        float bg_change = dx.segment<3>(9).norm() * 1000.0f;  // mrad/s
        float ba_change = dx.segment<3>(12).norm() * 100.0f;  // cm/s²
        
        spdlog::info("[Estimator] Iteration {}: rot={:.4f}deg, pos={:.4f}cm, vel={:.4f}cm/s, bg={:.4f}mrad/s, ba={:.4f}cm/s²",
                     iter, rot_change, pos_change, vel_change, bg_change, ba_change);
        
        // Convergence check: rotation < 0.01 deg, position < 0.015 cm
        if (rot_change < 0.01f && pos_change < 1.5f) {
            converged = true;
            spdlog::info("[Estimator] Converged after {} iterations", iter + 1);
            
            // Log current bias values
            spdlog::info("[Estimator] Current biases:");
            spdlog::info("  - Gyro bias: [{:.6f}, {:.6f}, {:.6f}] rad/s",
                        m_current_state.m_gyro_bias.x(), 
                        m_current_state.m_gyro_bias.y(), 
                        m_current_state.m_gyro_bias.z());
            spdlog::info("  - Acc bias: [{:.6f}, {:.6f}, {:.6f}] m/s²",
                        m_current_state.m_acc_bias.x(), 
                        m_current_state.m_acc_bias.y(), 
                        m_current_state.m_acc_bias.z());
            
            // Update covariance: P = (I - G) * P
            Eigen::Matrix<float, 18, 18> I18 = Eigen::Matrix<float, 18, 18>::Identity();
            Eigen::Matrix<float, 18, 18> P_updated = (I18 - G) * P_prior;
            m_current_state.m_covariance = P_updated;
            
            break;
        }
    }
    
    if (!converged) {
        spdlog::warn("[Estimator] Did not converge after {} iterations", max_iterations);
    }
    
    auto end_ikf = std::chrono::high_resolution_clock::now();
    double ikf_time = std::chrono::duration<double, std::milli>(end_ikf - start_ikf).count();
    
    auto end_total = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
    
    spdlog::warn("[UpdateWithLidar] ⏱️  IKF update took {:.2f} ms", ikf_time);
    spdlog::warn("[UpdateWithLidar] ⏱️  Total LiDAR update took {:.2f} ms", total_time);
}

// ============================================================================
// Correspondence Finding
// ============================================================================

std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float>> 
Estimator::FindCorrespondences(const PointCloudPtr scan) {
    std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float>> correspondences;
    
    // Check if map is empty
    if (!m_map_cloud || m_map_cloud->empty()) {
        spdlog::warn("[Estimator] Map is empty, no correspondences found");
        return correspondences;
    }
    
    if (!scan || scan->empty()) {
        spdlog::warn("[Estimator] Scan is empty, no correspondences found");
        return correspondences;
    }
    
    // Check if KdTree is available
    if (!m_map_kdtree) {
        spdlog::error("[Estimator] Map KdTree not built! This should not happen.");
        return correspondences;
    }
    
    // Get current state
    Eigen::Matrix3f R_wb = m_current_state.m_rotation;
    Eigen::Vector3f t_wb = m_current_state.m_position;
    
    const int K = 5;  // Number of neighbors for plane fitting
    const float max_neighbor_distance = 1.0f;  // Maximum distance for neighbors (meters)
    const float max_plane_distance = 0.1f;  // Maximum distance from point to fitted plane
    
    spdlog::info("[Estimator] Finding correspondences: {} scan points, {} map points", 
                 scan->size(), m_map_cloud->size());
    
    // PRE-TRANSFORM: Transform entire scan to world frame ONCE (major optimization)
    auto start_transform = std::chrono::high_resolution_clock::now();
    PointCloudPtr scan_world = std::make_shared<PointCloud>();
    scan_world->reserve(scan->size());
    
    // Build transformation matrix: T_world_lidar = T_world_body * T_body_lidar
    Eigen::Matrix4f T_wb = Eigen::Matrix4f::Identity();
    T_wb.block<3,3>(0,0) = R_wb;
    T_wb.block<3,1>(0,3) = t_wb;
    
    Eigen::Matrix4f T_il = Eigen::Matrix4f::Identity();
    T_il.block<3,3>(0,0) = Extrinsics::R_il;
    T_il.block<3,1>(0,3) = Extrinsics::t_il;
    
    Eigen::Matrix4f T_wl = T_wb * T_il;  // Combined transformation
    
    for (const auto& pt_scan : *scan) {
        Eigen::Vector4f pt_homo(pt_scan.x, pt_scan.y, pt_scan.z, 1.0f);
        Eigen::Vector4f pt_world_homo = T_wl * pt_homo;
        scan_world->push_back(pt_world_homo.x(), pt_world_homo.y(), pt_world_homo.z());
    }
    
    auto end_transform = std::chrono::high_resolution_clock::now();
    double transform_time = std::chrono::duration<double, std::milli>(end_transform - start_transform).count();
    spdlog::warn("[FindCorrespondences] ⏱️  Pre-transform took {:.2f} ms", transform_time);
    
    // 2. For each point in transformed scan, find correspondences
    int valid_correspondences = 0;
    int total_attempts = 0;
    
    for (size_t i = 0; i < scan_world->size(); ++i) {
        total_attempts++;
        
        const auto& pt_world = scan_world->at(i);
        
        // Already in world frame - no transform needed!
        Point3D query_point;
        query_point.x = pt_world.x;
        query_point.y = pt_world.y;
        query_point.z = pt_world.z;
        
        // 3. Find K nearest neighbors in map (use cached KdTree)
        std::vector<int> neighbor_indices(K);
        std::vector<float> neighbor_sq_distances(K);
        
        int found = m_map_kdtree->NearestKSearch(query_point, K, neighbor_indices, neighbor_sq_distances);
        
        if (found < K) {
            continue;  // Not enough neighbors
        }
        
        // Check distance threshold (use squared distance)
        if (neighbor_sq_distances[K-1] > max_neighbor_distance * max_neighbor_distance) {
            continue;  // Neighbors too far away
        }
        
        // 4. Collect neighbor points for plane fitting
        std::vector<Eigen::Vector3f> neighbor_points;
        neighbor_points.reserve(K);
        
        for (int i = 0; i < K; i++) {
            const auto& neighbor_pt = m_map_cloud->at(neighbor_indices[i]);
            neighbor_points.emplace_back(neighbor_pt.x, neighbor_pt.y, neighbor_pt.z);
        }
        
        // 5. Fit plane to neighbors using covariance method (SVD)
        // Compute centroid
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        for (const auto& pt : neighbor_points) {
            centroid += pt;
        }
        centroid /= static_cast<float>(K);
        
        // Compute covariance matrix
        Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
        for (const auto& pt : neighbor_points) {
            Eigen::Vector3f diff = pt - centroid;
            covariance += diff * diff.transpose();
        }
        covariance /= static_cast<float>(K);
        
        // SVD to find plane normal (smallest eigenvector)
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(covariance, Eigen::ComputeFullU);
        Eigen::Vector3f plane_normal = svd.matrixU().col(2);  // Smallest singular value
        
        // Ensure normal points towards sensor (optional, for consistency)
        Eigen::Vector3f p_world_vec(pt_world.x, pt_world.y, pt_world.z);
        if (plane_normal.dot(p_world_vec - centroid) > 0) {
            plane_normal = -plane_normal;
        }
        
        // 6. Validate plane fit quality
        // Check if all neighbors are close to the fitted plane
        bool plane_valid = true;
        for (const auto& pt : neighbor_points) {
            float dist_to_plane = std::abs(plane_normal.dot(pt - centroid));
            if (dist_to_plane > max_plane_distance) {
                plane_valid = false;
                break;
            }
        }
        
        if (!plane_valid) {
            continue;  // Plane fit is poor (points too scattered)
        }
        
        // Check planarity using singular values
        Eigen::Vector3f singular_values = svd.singularValues();
        float planarity = singular_values(2) / singular_values(0);  // Should be small for good plane
        
        if (planarity > 0.1f) {
            continue;  // Not planar enough
        }
        
        // 7. Add valid correspondence with plane information
        // Plane equation: n^T * x + d = 0, where d = -n^T * centroid
        // Point-to-plane distance: dis_to_plane = n^T * p_w + d
        float plane_d = -plane_normal.dot(centroid);
        
        // Store original lidar point (before transformation)
        const auto& pt_scan_original = scan->at(i);
        Eigen::Vector3f p_lidar(pt_scan_original.x, pt_scan_original.y, pt_scan_original.z);
        
        // Store: (p_lidar, plane_normal_world, plane_d)
        correspondences.emplace_back(p_lidar, plane_normal, plane_d);
        valid_correspondences++;
    }
    
    spdlog::info("[Estimator] Found {} valid correspondences out of {} scan points ({:.1f}%)",
                 valid_correspondences, total_attempts, 
                 100.0f * valid_correspondences / std::max(1, total_attempts));
    
    return correspondences;
}

// ============================================================================
// Local Map Management
// ============================================================================

void Estimator::UpdateLocalMap(const PointCloudPtr scan) {
    auto start_total = std::chrono::high_resolution_clock::now();
    
    Eigen::Matrix3f R_wb = m_current_state.m_rotation;
    Eigen::Vector3f t_wb = m_current_state.m_position;
    
    // ===== Keyframe Check =====
    bool is_keyframe = false;
    
    if (m_first_keyframe) {
        // First frame is always a keyframe
        is_keyframe = true;
        m_first_keyframe = false;
        m_last_keyframe_position = t_wb;
        m_last_keyframe_rotation = R_wb;
        spdlog::info("[Keyframe] 🔑 First keyframe");
    } else {
        // Compute relative transformation T_delta = T_last^-1 * T_current
        // T_last = [R_last, t_last; 0, 1]
        // T_current = [R_wb, t_wb; 0, 1]
        Eigen::Matrix4f T_last = Eigen::Matrix4f::Identity();
        T_last.block<3, 3>(0, 0) = m_last_keyframe_rotation;
        T_last.block<3, 1>(0, 3) = m_last_keyframe_position;
        
        Eigen::Matrix4f T_current = Eigen::Matrix4f::Identity();
        T_current.block<3, 3>(0, 0) = R_wb;
        T_current.block<3, 1>(0, 3) = t_wb;
        
        // T_delta = T_last^-1 * T_current
        Eigen::Matrix4f T_delta = T_last.inverse() * T_current;
        
        // Extract relative translation and rotation from T_delta
        Eigen::Vector3f t_delta = T_delta.block<3, 1>(0, 3);
        Eigen::Matrix3f R_delta = T_delta.block<3, 3>(0, 0);
        
        // Check translation: 1m threshold
        float translation = t_delta.norm();
        
        // Check rotation: 10 degree threshold
        Eigen::AngleAxisf angle_axis(R_delta);
        float rotation_deg = std::abs(angle_axis.angle()) * 180.0f / M_PI;
        
        if (translation >= 1.0f || rotation_deg >= 10.0f) {
            is_keyframe = true;
            m_last_keyframe_position = t_wb;
            m_last_keyframe_rotation = R_wb;
            spdlog::warn("[Keyframe] 🔑 New keyframe! Translation: {:.2f}m, Rotation: {:.2f}°", 
                         translation, rotation_deg);
        } else {
            spdlog::debug("[Keyframe] ⏭️  Skip (trans: {:.2f}m, rot: {:.2f}°)", 
                          translation, rotation_deg);
            
            // Skip map update but still rebuild KdTree and frustum filter
            auto start_frustum = std::chrono::high_resolution_clock::now();
            
            // Apply Frustum Culling to existing map
            Eigen::Matrix3f R_li = Extrinsics::R_il.transpose();
            Eigen::Vector3f t_li = -R_li * Extrinsics::t_il;
            Eigen::Matrix3f R_iw = R_wb.transpose();
            Eigen::Vector3f t_iw = -R_iw * t_wb;
            Eigen::Matrix3f R_lw = R_li * R_iw;
            Eigen::Vector3f t_lw = R_li * t_iw + t_li;
            
            auto frustum_filtered_map = std::make_shared<PointCloud>();
            FrustumFilter frustum_filter;
            frustum_filter.SetSensorPose(R_lw, t_lw);
            frustum_filter.SetFOV(90.0f, 90.0f);
            frustum_filter.SetMaxRange(50.0f);
            frustum_filter.SetInputCloud(m_map_cloud);
            frustum_filter.Filter(*frustum_filtered_map);
            
            auto end_frustum = std::chrono::high_resolution_clock::now();
            double frustum_time = std::chrono::duration<double, std::milli>(end_frustum - start_frustum).count();
            
            // Apply voxel downsampling
            auto start_voxel = std::chrono::high_resolution_clock::now();
            auto downsampled_cloud = std::make_shared<PointCloud>();
            VoxelGrid voxel_filter;
            voxel_filter.SetInputCloud(frustum_filtered_map);
            voxel_filter.SetLeafSize(0.4f);
            voxel_filter.Filter(*downsampled_cloud);
            auto end_voxel = std::chrono::high_resolution_clock::now();
            double voxel_time = std::chrono::duration<double, std::milli>(end_voxel - start_voxel).count();
            
            m_map_cloud = downsampled_cloud;
            
            // Rebuild KdTree
            auto start_kdtree = std::chrono::high_resolution_clock::now();
            if (!m_map_cloud->empty()) {
                m_map_kdtree = std::make_shared<KdTree>();
                m_map_kdtree->SetInputCloud(m_map_cloud);
            }
            auto end_kdtree = std::chrono::high_resolution_clock::now();
            double kdtree_time = std::chrono::duration<double, std::milli>(end_kdtree - start_kdtree).count();
            
            auto end_total = std::chrono::high_resolution_clock::now();
            double total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
            
            spdlog::warn("[UpdateLocalMap] ⏭️  Non-keyframe: Frustum: {:.2f} ms, Voxel: {:.2f} ms, KdTree: {:.2f} ms, Total: {:.2f} ms (Map: {} pts)",
                         frustum_time, voxel_time, kdtree_time, total_time, m_map_cloud->size());
            return;
        }
    }
    
    // ===== Add new scan to map (only for keyframes) =====
    auto start_transform = std::chrono::high_resolution_clock::now();
    
    int added_count = 0;
    for (const auto& pt : *scan) {
        // LiDAR point in sensor frame
        Eigen::Vector3f p_lidar(pt.x, pt.y, pt.z);
        
        // Transform: p_world = R_wb * (R_il * p_lidar + t_il) + t_wb
        Eigen::Vector3f p_imu = Extrinsics::R_il * p_lidar + Extrinsics::t_il;
        Eigen::Vector3f p_world = R_wb * p_imu + t_wb;
        
        // Add to map cloud
        Point3D map_pt;
        map_pt.x = p_world.x();
        map_pt.y = p_world.y();
        map_pt.z = p_world.z();
        map_pt.intensity = pt.intensity;
        map_pt.offset_time = pt.offset_time;
        m_map_cloud->push_back(map_pt);
        added_count++;
    }
    auto end_transform = std::chrono::high_resolution_clock::now();
    double transform_time = std::chrono::duration<double, std::milli>(end_transform - start_transform).count();

    spdlog::info("[Estimator] Added {} points to map, total before filtering: {}", added_count, m_map_cloud->size());

    // ===== Apply Frustum Culling to Map =====
    auto start_frustum = std::chrono::high_resolution_clock::now();
    // Keep only points within current sensor FOV and range
    // T_lw = T_li * T_iw = T_li * T_wb^-1
    Eigen::Matrix3f R_li = Extrinsics::R_il.transpose();  // LiDAR to IMU inverse
    Eigen::Vector3f t_li = -R_li * Extrinsics::t_il;
    Eigen::Matrix3f R_iw = R_wb.transpose();  // IMU to world inverse
    Eigen::Vector3f t_iw = -R_iw * t_wb;
    
    // Compose: R_lw = R_li * R_iw, t_lw = R_li * t_iw + t_li
    Eigen::Matrix3f R_lw = R_li * R_iw;
    Eigen::Vector3f t_lw = R_li * t_iw + t_li;
    
    auto frustum_filtered_map = std::make_shared<PointCloud>();
    FrustumFilter frustum_filter;
    frustum_filter.SetSensorPose(R_lw, t_lw);  // World to LiDAR transform
    frustum_filter.SetFOV(90.0f, 90.0f);  // ±45° horizontal and vertical
    frustum_filter.SetMaxRange(50.0f);     // 50m max range
    frustum_filter.SetInputCloud(m_map_cloud);
    frustum_filter.Filter(*frustum_filtered_map);
    auto end_frustum = std::chrono::high_resolution_clock::now();
    double frustum_time = std::chrono::duration<double, std::milli>(end_frustum - start_frustum).count();
    
    spdlog::info("[Estimator] Frustum culling: {} → {} points ({:.1f}%)",
                 m_map_cloud->size(), frustum_filtered_map->size(),
                 100.0f * frustum_filtered_map->size() / std::max(1, static_cast<int>(m_map_cloud->size())));

    // Apply voxel downsampling to control map size
    auto start_voxel = std::chrono::high_resolution_clock::now();
    auto downsampled_cloud = std::make_shared<PointCloud>();

    VoxelGrid voxel_filter;
    voxel_filter.SetInputCloud(frustum_filtered_map);  // Downsample frustum-filtered map
    voxel_filter.SetLeafSize(0.4f); // 40cm voxel size
    voxel_filter.Filter(*downsampled_cloud);
    auto end_voxel = std::chrono::high_resolution_clock::now();
    double voxel_time = std::chrono::duration<double, std::milli>(end_voxel - start_voxel).count();

    m_map_cloud = downsampled_cloud;
    spdlog::info("[Estimator] Map downsampled to {} points", m_map_cloud->size());

    // Rebuild KdTree after map update
    auto start_kdtree = std::chrono::high_resolution_clock::now();
    if (!m_map_cloud->empty())
    {
        m_map_kdtree = std::make_shared<KdTree>();
        m_map_kdtree->SetInputCloud(m_map_cloud);
        spdlog::debug("[Estimator] KdTree rebuilt with {} points", m_map_cloud->size());
    }
    auto end_kdtree = std::chrono::high_resolution_clock::now();
    double kdtree_time = std::chrono::duration<double, std::milli>(end_kdtree - start_kdtree).count();
    
    auto end_total = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();

    spdlog::info("[Estimator] Map updated: {} points total", m_map_cloud->size());
    spdlog::warn("[UpdateLocalMap] ⏱️  Transform: {:.2f} ms, Frustum: {:.2f} ms, Voxel: {:.2f} ms, KdTree: {:.2f} ms, Total: {:.2f} ms",
                 transform_time, frustum_time, voxel_time, kdtree_time, total_time);
}

void Estimator::CleanLocalMap() {
    // TODO: Implement proper map cleaning
    // For now, just limit the size by removing oldest points
    
    size_t map_size = m_map_cloud->size();
    size_t max_size = static_cast<size_t>(m_params.max_map_points);
    
    if (map_size > max_size) {
        // Create new cloud with recent points
        auto new_cloud = std::make_shared<PointCloud>();
        int start_idx = map_size - max_size;
        int idx = 0;
        
        for (const auto& pt : *m_map_cloud) {
            if (idx >= start_idx) {
                new_cloud->push_back(pt);
            }
            idx++;
        }
        
        m_map_cloud = new_cloud;
        
        // Rebuild KdTree after cleaning
        if (!m_map_cloud->empty()) {
            m_map_kdtree = std::make_shared<KdTree>();
            m_map_kdtree->SetInputCloud(m_map_cloud);
            spdlog::debug("[Estimator] KdTree rebuilt after cleaning");
        }
        
        spdlog::info("[Estimator] Map cleaned: {} points", m_map_cloud->size());
    }
}

// ============================================================================
// Jacobian Computation
// ============================================================================

void Estimator::ComputeLidarJacobians(
    const std::vector<std::tuple<Eigen::Vector3f, Eigen::Vector3f, float>>& correspondences,
    Eigen::MatrixXf& H,
    Eigen::VectorXf& residual) 
{
    // Compute Jacobian matrix H and residual vector for point-to-plane correspondences
    // State: [rotation(3), position(3), velocity(3), gyro_bias(3), acc_bias(3), gravity(3)]
    // LiDAR only observes rotation and position, other states have zero Jacobian
    
    int num_corr = correspondences.size();
    H.resize(num_corr, 18);
    residual.resize(num_corr);
    
    H.setZero();
    residual.setZero();
    
    // Get current state
    Eigen::Matrix3f R_wb = m_current_state.m_rotation;
    Eigen::Vector3f t_wb = m_current_state.m_position;
    
    // Process each correspondence
    for (int i = 0; i < num_corr; i++) {
        // Extract correspondence data: (p_lidar, plane_normal, plane_d)
        const Eigen::Vector3f& p_lidar = std::get<0>(correspondences[i]);
        const Eigen::Vector3f& norm_vec = std::get<1>(correspondences[i]);  // plane normal (world frame)
        const float plane_d = std::get<2>(correspondences[i]);
        
        // Transform point through chain: LiDAR -> IMU -> World
        // p_imu = R_il * p_lidar + t_il
        Eigen::Vector3f p_imu = Extrinsics::R_il * p_lidar + Extrinsics::t_il;
        
        // p_world = R_wb * p_imu + t_wb
        Eigen::Vector3f p_world = R_wb * p_imu + t_wb;
        
        // ===== Residual Computation =====
        // Point-to-plane distance: dis_to_plane = n^T * p_w + d
        // Measurement vector: meas_vec(i) = -dis_to_plane
        // Therefore: residual = -(n^T * p_world + d)
        residual(i) = -(norm_vec.dot(p_world) + plane_d);
        
        // ===== Jacobian Computation =====
        
        // Transform normal to body frame: C = R_wb^T * n
        Eigen::Vector3f C = R_wb.transpose() * norm_vec;
        
        // Rotation Jacobian: A = [p_imu]× * C
        // A = point_crossmat * state_rotation.transpose() * normal
        // Using POSITIVE sign for proper gradient direction
        Eigen::Matrix3f p_imu_skew = Hat(p_imu);
        Eigen::Vector3f A = p_imu_skew * C;
        
        // Position Jacobian: simply the normal vector
        // ∂r/∂t = ∂(n^T * (R * p_imu + t))/∂t = n^T
        
        // Fill Jacobian row (1×18)
        // State order: [rotation(3), position(3), velocity(3), gyro_bias(3), acc_bias(3), gravity(3)]
        H.block<1, 3>(i, 0) = A.transpose();           // ∂r/∂rotation
        H.block<1, 3>(i, 3) = norm_vec.transpose();    // ∂r/∂position
        // H.block<1, 12>(i, 6) = 0;                   // velocity, biases, gravity (already zero)
    }
}

// ============================================================================
// Noise Updates
// ============================================================================

void Estimator::UpdateProcessNoise(double dt) {
    // Scale noise by time step (already set in constructor)
    // Q matrix is used as Q * dt in propagation
}

void Estimator::UpdateMeasurementNoise(int num_correspondences) {
    // Measurement noise R is diagonal (independent residuals)
    m_measurement_noise = Eigen::MatrixXf::Identity(num_correspondences, num_correspondences);
    m_measurement_noise *= m_params.lidar_noise_std * m_params.lidar_noise_std;
}

void Estimator::ApplyStateCorrection(const Eigen::VectorXf& dx) {
    // Apply state correction on manifold (IEKF update)
    // State: [rotation(3), position(3), velocity(3), gyro_bias(3), acc_bias(3), gravity(3)]
    
    if (dx.size() != 18) {
        spdlog::error("[Estimator] Invalid state correction size: {} (expected 18)", dx.size());
        return;
    }
    
    // 1. Rotation: R_new = R * Exp(δθ)  (right perturbation on SO(3))
    Eigen::Vector3f dtheta = dx.segment<3>(0);
    Eigen::Matrix3f dR = SO3::Exp(dtheta).Matrix();
    m_current_state.m_rotation = m_current_state.m_rotation * dR;
    
    // 2. Position: p_new = p + δp  (additive in R^3)
    m_current_state.m_position += dx.segment<3>(3);
    
    // 3. Velocity: v_new = v + δv  (additive in R^3)
    m_current_state.m_velocity += dx.segment<3>(6);
    
    // 4. Gyroscope bias: bg_new = bg + δbg  (additive in R^3)
    m_current_state.m_gyro_bias += dx.segment<3>(9);
    
    // 5. Accelerometer bias: ba_new = ba + δba  (additive in R^3)
    m_current_state.m_acc_bias += dx.segment<3>(12);
    
    // 6. Gravity: g_new = g + δg  (additive in R^3)
    m_current_state.m_gravity += dx.segment<3>(15);
    
    // Log correction magnitude for debugging
    spdlog::debug("[Estimator] State correction applied: rotation={:.6f}, position={:.6f}, velocity={:.6f}",
                  dtheta.norm(), dx.segment<3>(3).norm(), dx.segment<3>(6).norm());
}

// ============================================================================
// State Getters
// ============================================================================

State Estimator::GetCurrentState() const {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    return m_current_state;
}

std::vector<State> Estimator::GetTrajectory() const {
    std::lock_guard<std::mutex> lock(m_state_mutex);
    return std::vector<State>(m_trajectory.begin(), m_trajectory.end());
}

Estimator::Statistics Estimator::GetStatistics() const {
    std::lock_guard<std::mutex> lock(m_stats_mutex);
    return m_statistics;
}

// ============================================================================
// Undistortion & Interpolation (Placeholder)
// ============================================================================

PointCloudPtr Estimator::UndistortPointCloud(
    const PointCloudPtr cloud,
    double scan_start_time,
    double scan_end_time) 
{
    // TODO: Implement motion undistortion
    // 1. For each point with offset_time:
    //    - Interpolate state at (scan_start_time + offset_time)
    //    - Transform point using interpolated state
    
    return cloud;
}

State Estimator::InterpolateState(double timestamp) const {
    // TODO: Implement state interpolation using IMU buffer
    // Linear interpolation between nearest IMU measurements
    
    return m_current_state;
}

// ============================================================================
// Feature Extraction (Placeholder)
// ============================================================================

void Estimator::ExtractPlanarFeatures(
    const PointCloudPtr cloud,
    std::vector<MapPoint>& features) 
{
    // TODO: Implement planar feature extraction
    // For each point, check local neighborhood planarity
    
    features.clear();
}

} // namespace lio
