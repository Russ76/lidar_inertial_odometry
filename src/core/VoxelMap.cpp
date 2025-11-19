/**
 * @file      VoxelMap.cpp
 * @brief     Implementation of voxel-based hash map for efficient nearest neighbor search
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#include "VoxelMap.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace lio {

VoxelMap::VoxelMap(float voxel_size) 
    : m_voxel_size(voxel_size), m_max_hit_count(10) {
}

void VoxelMap::SetVoxelSize(float size) {
    if (size <= 0.0f) {
        throw std::invalid_argument("Voxel size must be positive");
    }
    
    // If size changed, need to rebuild the map
    if (std::abs(m_voxel_size - size) > 1e-6f) {
        m_voxel_size = size;
        
        // Rebuild map with new voxel size
        std::vector<Point3D> points_copy = m_all_points;
        Clear();
        for (const auto& pt : points_copy) {
            AddPoint(pt);
        }
    }
}

VoxelKey VoxelMap::PointToVoxelKey(const Point3D& point) const {
    int vx = static_cast<int>(std::floor(point.x / m_voxel_size));
    int vy = static_cast<int>(std::floor(point.y / m_voxel_size));
    int vz = static_cast<int>(std::floor(point.z / m_voxel_size));
    return VoxelKey(vx, vy, vz);
}

void VoxelMap::AddPoint(const Point3D& point) {
    // Get voxel key for this point
    VoxelKey key = PointToVoxelKey(point);
    
    // Add point to global storage
    int global_index = static_cast<int>(m_all_points.size());
    m_all_points.push_back(point);
    
    // Get or create voxel data
    VoxelData& voxel_data = m_voxel_map[key];
    
    // Update centroid with weighted average
    Eigen::Vector3f point_vec(point.x, point.y, point.z);
    int n = voxel_data.point_indices.size();
    
    if (n == 0) {
        // First point in voxel
        voxel_data.centroid = point_vec;
        voxel_data.hit_count = 1;
    } else {
        // Weighted average: new_centroid = (n * old_centroid + new_point) / (n + 1)
        voxel_data.centroid = (voxel_data.centroid * n + point_vec) / (n + 1);
    }
    
    // Add index to voxel's point list
    voxel_data.point_indices.push_back(global_index);
}

void VoxelMap::AddPointCloud(const PointCloudPtr& cloud) {
    if (!cloud || cloud->empty()) {
        return;
    }
    
    // Reserve space for efficiency
    m_all_points.reserve(m_all_points.size() + cloud->size());
    
    for (size_t i = 0; i < cloud->size(); ++i) {
        AddPoint(cloud->at(i));
    }
}

std::vector<VoxelKey> VoxelMap::GetNeighborVoxels(const VoxelKey& center) const {
    std::vector<VoxelKey> neighbors;
    neighbors.reserve(27);  // 3x3x3 = 27 voxels
    
    // Search in 3x3x3 grid around center voxel
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dz = -1; dz <= 1; ++dz) {
                neighbors.emplace_back(center.x + dx, center.y + dy, center.z + dz);
            }
        }
    }
    
    return neighbors;
}

int VoxelMap::FindKNearestNeighbors(const Point3D& query_point, 
                                     int K,
                                     std::vector<int>& indices,
                                     std::vector<float>& squared_distances) {
    indices.clear();
    squared_distances.clear();
    
    if (K <= 0 || m_all_points.empty()) {
        return 0;
    }
    
    // Get query voxel key
    VoxelKey query_voxel = PointToVoxelKey(query_point);
    
    // Get all 27 neighboring voxels
    std::vector<VoxelKey> neighbor_voxels = GetNeighborVoxels(query_voxel);
    
    // Collect candidate centroids from neighboring voxels
    struct Candidate {
        VoxelKey voxel_key;
        Eigen::Vector3f centroid;
        float squared_distance;
        
        bool operator<(const Candidate& other) const {
            return squared_distance < other.squared_distance;
        }
    };
    
    std::vector<Candidate> candidates;
    Eigen::Vector3f query_vec(query_point.x, query_point.y, query_point.z);
    
    for (const auto& voxel_key : neighbor_voxels) {
        auto it = m_voxel_map.find(voxel_key);
        if (it == m_voxel_map.end() || it->second.point_indices.empty()) {
            continue;  // Voxel is empty
        }
        
        // Use weighted centroid for distance calculation
        const Eigen::Vector3f& centroid = it->second.centroid;
        float sq_dist = (query_vec - centroid).squaredNorm();
        
        candidates.push_back({voxel_key, centroid, sq_dist});
    }
    
    // If no candidates found, return 0
    if (candidates.empty()) {
        return 0;
    }
    
    // Sort candidates by distance to centroid
    std::partial_sort(candidates.begin(), 
                     candidates.begin() + std::min(K, static_cast<int>(candidates.size())),
                     candidates.end());
    
    // Extract K nearest neighbors
    int num_found = std::min(K, static_cast<int>(candidates.size()));
    indices.reserve(num_found);
    squared_distances.reserve(num_found);
    
    for (int i = 0; i < num_found; ++i) {
        // Return first point index from the voxel (representative point)
        const VoxelData& voxel_data = m_voxel_map[candidates[i].voxel_key];
        if (!voxel_data.point_indices.empty()) {
            indices.push_back(voxel_data.point_indices[0]);
            squared_distances.push_back(candidates[i].squared_distance);
        }
    }
    
    return indices.size();
}

void VoxelMap::Clear() {
    m_voxel_map.clear();
    m_all_points.clear();
}

void VoxelMap::UpdateVoxelMap(const PointCloudPtr& new_cloud,
                               const Eigen::Vector3d& sensor_position,
                               double max_distance, bool is_keyframe) {
    if (new_cloud->empty()) return;
    
    // Step 1: Mark voxels that are hit by the new scan
    // For each point in the scan, mark its voxel and neighboring voxels as "hit"
    std::unordered_set<VoxelKey, VoxelKeyHash> hit_voxels_set;
    
    for (const auto& pt : *new_cloud) {
        VoxelKey center_key = PointToVoxelKey(pt);
        
        // Mark the voxel containing this point
        hit_voxels_set.insert(center_key);
        
        // Also mark 26 neighboring voxels (3x3x3 - 1)
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    VoxelKey neighbor_key(center_key.x + dx, 
                                         center_key.y + dy, 
                                         center_key.z + dz);
                    hit_voxels_set.insert(neighbor_key);
                }
            }
        }
    }
    
    // Step 2: Update hit counts for all existing voxels
    std::vector<VoxelKey> voxels_to_remove;
    
    for (auto& pair : m_voxel_map) {
        const VoxelKey& key = pair.first;
        VoxelData& voxel_data = pair.second;
        
        // Check if this voxel was hit by the current scan
        if (hit_voxels_set.find(key) != hit_voxels_set.end()) {
            // Voxel is hit by current scan -> increment hit count (up to max)
            if (voxel_data.hit_count < m_max_hit_count) {
                voxel_data.hit_count++;
            }
        } else {
            // Voxel is not hit -> decrement hit count by 1
            voxel_data.hit_count--;
            
            // Mark for removal if hit count drops below 1
            if (voxel_data.hit_count < 1) {
                voxels_to_remove.push_back(key);
            }
        }
    }
    
    // Step 3: Remove voxels with hit_count < 1
    for (const auto& key : voxels_to_remove) {
        m_voxel_map.erase(key);
    }
    
    // Step 4: Add new points (creates new voxels with hit_count=1, updates centroids)
    if(is_keyframe) 
    {
        AddPointCloud(new_cloud);
    }
}

std::vector<VoxelKey> VoxelMap::GetOccupiedVoxels() const {
    std::vector<VoxelKey> occupied_voxels;
    occupied_voxels.reserve(m_voxel_map.size());
    
    for (const auto& pair : m_voxel_map) {
        if (!pair.second.point_indices.empty()) {  // Only add non-empty voxels
            occupied_voxels.push_back(pair.first);
        }
    }
    
    return occupied_voxels;
}

Eigen::Vector3f VoxelMap::VoxelKeyToCenter(const VoxelKey& key) const {
    // Convert voxel key back to world coordinates (center of voxel)
    float center_x = (key.x + 0.5f) * m_voxel_size;
    float center_y = (key.y + 0.5f) * m_voxel_size;
    float center_z = (key.z + 0.5f) * m_voxel_size;
    
    return Eigen::Vector3f(center_x, center_y, center_z);
}

Eigen::Vector3f VoxelMap::GetVoxelCentroid(const VoxelKey& key) const {
    auto it = m_voxel_map.find(key);
    if (it == m_voxel_map.end()) {
        // Voxel not found, return geometric center as fallback
        return VoxelKeyToCenter(key);
    }
    
    // Return the weighted centroid stored in VoxelData
    return it->second.centroid;
}

int VoxelMap::GetVoxelHitCount(const VoxelKey& key) const {
    auto it = m_voxel_map.find(key);
    if (it == m_voxel_map.end()) {
        return 0;  // Voxel not found
    }
    
    return it->second.hit_count;
}

void VoxelMap::MarkVoxelAsHit(const VoxelKey& key) {
    m_hit_voxels[key] = true;
}

void VoxelMap::ClearHitMarkers() {
    m_hit_voxels.clear();
}

bool VoxelMap::IsVoxelHit(const VoxelKey& key) const {
    return m_hit_voxels.find(key) != m_hit_voxels.end();
}

std::vector<VoxelKey> VoxelMap::GetHitVoxels() const {
    std::vector<VoxelKey> hit_voxels;
    hit_voxels.reserve(m_hit_voxels.size());
    
    for (const auto& pair : m_hit_voxels) {
        hit_voxels.push_back(pair.first);
    }
    
    return hit_voxels;
}

} // namespace lio
