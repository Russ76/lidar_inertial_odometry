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
    
    // 2. Compute mean acceleration and gyroscope (FAST-LIO style running average)
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
    
    spdlog::info("  Acc bias: [{:.4f}, {:.4f}, {:.4f}] m/s²",
                 acc_bias_estimate.x(), acc_bias_estimate.y(), acc_bias_estimate.z());
    
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
    
    // First frame: initialize map
    if (m_first_lidar_frame) {
        spdlog::info("[Estimator] First LiDAR frame - initializing map");
        UpdateLocalMap(lidar.cloud);
        m_first_lidar_frame = false;
        m_last_lidar_time = lidar.timestamp;
        m_last_lidar_state = m_current_state;
        m_frame_count++;
        return;
    }
    
    // Motion check: skip if not enough motion
    float distance = (m_current_state.m_position - m_last_lidar_state.m_position).norm();
    if (distance < m_params.min_motion_threshold) {
        spdlog::info("[Estimator] Skipping frame (insufficient motion: {:.3f} m)", distance);
        return;
    }
    
    // Undistort point cloud using IMU integration
    PointCloudPtr undistorted_cloud = lidar.cloud;
    if (m_params.enable_undistortion) {
        // TODO: Implement motion undistortion using IMU buffer
        // For now, skip undistortion
    }
    
    // Downsample point cloud
    // TODO: Implement voxel downsampling
    PointCloudPtr scan = undistorted_cloud;
    
    // Iterated Kalman Filter Update
    UpdateWithLidar(lidar);
    
    // Update local map with new scan
    UpdateLocalMap(scan);
    
    // Clean old points from map
    CleanLocalMap();
    
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
    // TODO: Implement Iterated Kalman Filter update
    // 1. Transform scan to world frame using current state
    // 2. Find correspondences (5 nearest neighbors per point)
    // 3. Fit planes to neighbors
    // 4. Compute point-to-plane residuals
    // 5. Compute Jacobians H
    // 6. Kalman update: K = P*H^T*(H*P*H^T + R)^-1, x = x + K*r
    // 7. Iterate 4-5 times
    
    spdlog::info("[Estimator] LiDAR update (TODO: implement Iterated Kalman)");
}

// ============================================================================
// Correspondence Finding
// ============================================================================

std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> 
Estimator::FindCorrespondences(const PointCloudPtr scan) {
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> correspondences;
    
    // TODO: Implement correspondence search
    // 1. Build KdTree from map
    // 2. For each point in scan:
    //    - Transform to world frame
    //    - Find 5 nearest neighbors
    //    - Fit plane to neighbors
    //    - Add (point, plane_normal) to correspondences
    
    return correspondences;
}

// ============================================================================
// Local Map Management
// ============================================================================

void Estimator::UpdateLocalMap(const PointCloudPtr scan) {
    // Transform scan to world frame
    Eigen::Matrix3f R = m_current_state.m_rotation;
    Eigen::Vector3f t = m_current_state.m_position;
    
    int added_count = 0;
    for (const auto& pt : *scan) {
        // LiDAR point in sensor frame
        Eigen::Vector3f p_lidar(pt.x, pt.y, pt.z);
        
        // Transform: p_world = R_wb * (R_il * p_lidar + t_il) + t_wb
        Eigen::Vector3f p_imu = Extrinsics::R_il * p_lidar + Extrinsics::t_il;
        Eigen::Vector3f p_world = R * p_imu + t;
        
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
    
    spdlog::info("[Estimator] Map updated: {} points (added {})", m_map_cloud->size(), added_count);
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
        spdlog::info("[Estimator] Map cleaned: {} points", m_map_cloud->size());
    }
}

// ============================================================================
// Jacobian Computation
// ============================================================================

void Estimator::ComputeLidarJacobians(
    const std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>>& correspondences,
    Eigen::MatrixXf& H,
    Eigen::VectorXf& residual) 
{
    // TODO: Implement Jacobian computation
    // For each correspondence (point, plane_normal):
    //   residual = n^T * (R * (R_il * p_lidar + t_il) + t - plane_point)
    //   dR/dtheta = [point]_x (rotation perturbation)
    //   dt/dt = I (translation perturbation)
    
    int num_corr = correspondences.size();
    H.resize(num_corr, 18);
    residual.resize(num_corr);
    
    H.setZero();
    residual.setZero();
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
