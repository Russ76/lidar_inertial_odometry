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

namespace lio {

VoxelMap::VoxelMap(float voxel_size) 
    : m_voxel_size(voxel_size) {
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
    
    // Add index to voxel's point list
    m_voxel_map[key].push_back(global_index);
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
    
    // Collect candidate points from neighboring voxels
    struct Candidate {
        int index;
        float squared_distance;
        
        bool operator<(const Candidate& other) const {
            return squared_distance < other.squared_distance;
        }
    };
    
    std::vector<Candidate> candidates;
    
    for (const auto& voxel_key : neighbor_voxels) {
        auto it = m_voxel_map.find(voxel_key);
        if (it == m_voxel_map.end()) {
            continue;  // Voxel is empty
        }
        
        // Check all points in this voxel
        for (int point_idx : it->second) {
            const Point3D& candidate_point = m_all_points[point_idx];
            float sq_dist = query_point.squared_distance_to(candidate_point);
            
            candidates.push_back({point_idx, sq_dist});
        }
    }
    
    // If no candidates found, return 0
    if (candidates.empty()) {
        return 0;
    }
    
    // Sort candidates by distance
    std::partial_sort(candidates.begin(), 
                     candidates.begin() + std::min(K, static_cast<int>(candidates.size())),
                     candidates.end());
    
    // Extract K nearest neighbors
    int num_found = std::min(K, static_cast<int>(candidates.size()));
    indices.reserve(num_found);
    squared_distances.reserve(num_found);
    
    for (int i = 0; i < num_found; ++i) {
        indices.push_back(candidates[i].index);
        squared_distances.push_back(candidates[i].squared_distance);
    }
    
    return num_found;
}

void VoxelMap::Clear() {
    m_voxel_map.clear();
    m_all_points.clear();
}

std::vector<VoxelKey> VoxelMap::GetOccupiedVoxels() const {
    std::vector<VoxelKey> occupied_voxels;
    occupied_voxels.reserve(m_voxel_map.size());
    
    for (const auto& pair : m_voxel_map) {
        if (!pair.second.empty()) {  // Only add non-empty voxels
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
