/**
 * @file      VoxelMap.h
 * @brief     Voxel-based hash map for efficient nearest neighbor search
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-18
 * @copyright Copyright (c) 2025 Seungwon Choi. All rights reserved.
 *
 * @par License
 * This project is released under the MIT License.
 */

#ifndef VOXEL_MAP_H
#define VOXEL_MAP_H

#include "PointCloudUtils.h"
#include <unordered_map>
#include <vector>
#include <Eigen/Dense>

namespace lio {

/**
 * @brief Voxel key for spatial hashing
 * Represents a 3D grid cell by integer indices (x, y, z)
 */
struct VoxelKey {
    int x, y, z;
    
    VoxelKey() : x(0), y(0), z(0) {}
    VoxelKey(int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
    
    bool operator==(const VoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
    
    // For debugging
    std::string ToString() const {
        return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
    }
};

/**
 * @brief Hash function for VoxelKey to use in unordered_map
 */
struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& key) const {
        // Cantor pairing function for combining hash values
        std::size_t h1 = std::hash<int>{}(key.x);
        std::size_t h2 = std::hash<int>{}(key.y);
        std::size_t h3 = std::hash<int>{}(key.z);
        
        // Combine hashes using XOR and bit shifting
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

/**
 * @brief Voxel-based spatial hash map for efficient nearest neighbor search
 * 
 * This data structure provides O(1) voxel lookup and fast K-nearest neighbor search
 * by checking only neighboring voxels (27 voxels in 3x3x3 grid).
 * 
 * Performance comparison vs KdTree:
 * - KdTree: O(N * log(M)) where N=query points, M=map size
 * - VoxelMap: O(N * K) where K=fixed (27 voxels * points_per_voxel)
 * 
 * Expected speedup: 3-5x faster for large maps (20k+ points)
 */
class VoxelMap {
public:
    /**
     * @brief Constructor
     * @param voxel_size Size of each voxel in meters (default: 0.5m)
     */
    explicit VoxelMap(float voxel_size = 0.5f);
    
    /**
     * @brief Set voxel size
     * @param size Voxel size in meters
     */
    void SetVoxelSize(float size);
    
    /**
     * @brief Get current voxel size
     */
    float GetVoxelSize() const { return m_voxel_size; }
    
    /**
     * @brief Add a point to the voxel map
     * @param point 3D point to add
     */
    void AddPoint(const Point3D& point);
    
    /**
     * @brief Add multiple points to the voxel map
     * @param cloud Point cloud to add
     */
    void AddPointCloud(const PointCloudPtr& cloud);
    
    /**
     * @brief Find K nearest neighbors using voxel-based search
     * @param query_point Query point
     * @param K Number of neighbors to find
     * @param indices Output: indices of nearest neighbors (relative to insertion order)
     * @param squared_distances Output: squared distances to neighbors
     * @return Number of neighbors found (may be less than K)
     */
    int FindKNearestNeighbors(const Point3D& query_point, 
                              int K,
                              std::vector<int>& indices,
                              std::vector<float>& squared_distances);
    
    /**
     * @brief Clear all points from the map
     */
    void Clear();
    
    /**
     * @brief Get total number of points in the map
     */
    size_t GetPointCount() const { return m_all_points.size(); }
    
    /**
     * @brief Get number of occupied voxels
     */
    size_t GetVoxelCount() const { return m_voxel_map.size(); }
    
    /**
     * @brief Get point by global index
     */
    const Point3D& GetPoint(int index) const { return m_all_points[index]; }
    
    /**
     * @brief Get all occupied voxel keys for visualization
     * @return Vector of all occupied voxel keys
     */
    std::vector<VoxelKey> GetOccupiedVoxels() const;
    
    /**
     * @brief Convert voxel key to center position
     * @param key Voxel key
     * @return Center position of the voxel in world coordinates
     */
    Eigen::Vector3f VoxelKeyToCenter(const VoxelKey& key) const;
    
    /**
     * @brief Mark a voxel as hit by current scan
     * @param key Voxel key to mark
     */
    void MarkVoxelAsHit(const VoxelKey& key);
    
    /**
     * @brief Clear all hit markers
     */
    void ClearHitMarkers();
    
    /**
     * @brief Check if a voxel is hit by current scan
     * @param key Voxel key to check
     * @return True if voxel is hit
     */
    bool IsVoxelHit(const VoxelKey& key) const;
    
    /**
     * @brief Get all hit voxel keys
     * @return Vector of hit voxel keys
     */
    std::vector<VoxelKey> GetHitVoxels() const;
    
private:
    /**
     * @brief Convert 3D point to voxel key
     */
    VoxelKey PointToVoxelKey(const Point3D& point) const;
    
    /**
     * @brief Get all 27 neighboring voxel keys (3x3x3 grid including center)
     */
    std::vector<VoxelKey> GetNeighborVoxels(const VoxelKey& center) const;
    
    // ===== Member Variables =====
    
    float m_voxel_size;  ///< Size of each voxel in meters
    
    /// Voxel hash map: VoxelKey -> list of point indices
    std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> m_voxel_map;
    
    /// Storage for all points (indexed by global index)
    std::vector<Point3D> m_all_points;
    
    /// Hit markers for current scan visualization
    std::unordered_map<VoxelKey, bool, VoxelKeyHash> m_hit_voxels;
};

} // namespace lio

#endif // VOXEL_MAP_H
