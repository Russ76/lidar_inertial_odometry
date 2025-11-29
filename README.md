# Lidar-Inertial Odometry

### MIT License

## Features

- **Iterated Extended Kalman Filter (IEKF)**: Direct LiDAR-IMU fusion with nested iteration for re-linearization and convergence
- **Adaptive Robust Estimation**: Probabilistic Kernel Optimization (PKO) for automatic Huber loss scale tuning
- **Incremental Hierarchical Voxel Map**: 2-level hash-based spatial indexing (L0/L1) with occupied-only tracking for fast local KNN search
- **Pre-computed Surfel Planes**: L1 voxels store fitted plane surfels (normal, centroid, covariance) computed via SVD, enabling O(1) correspondence finding without per-point KNN/SVD
- **Motion compensation**: IMU-based undistortion for moving LiDAR scans

### Probabilistic Kernel Optimization (PKO)

This project implements adaptive robust estimation using Probabilistic Kernel Optimization for automatic Huber loss scale tuning. If you use this method in your research, please cite:

```bibtex
@article{choi2025pko,
  title={Probabilistic Kernel Optimization for Robust State Estimation},
  author={Choi, Seungwon and Kim, Tae-Wan},
  journal={IEEE Robotics and Automation Letters},
  volume={10},
  number={3},
  pages={2998--3005},
  year={2025},
  publisher={IEEE}
}
```

## Demo

[![LIO Demo](https://img.youtube.com/vi/2JymC0LWDWI/0.jpg)](https://www.youtube.com/watch?v=2JymC0LWDWI)



### Installation (Ubuntu 20.04)

```bash
cd lidar_inertial_odometry
./build.sh
```

This will:
1. Build Pangolin from `thirdparty/pangolin`
2. Build the main project with CMake

### Quick Start

#### M3DGR Dataset (Recommended)

**Download Pre-processed Dataset**:
- **Google Drive**: [M3DGR Parsed Dataset](https://drive.google.com/drive/folders/1zOmvw3sCwRQ0LHo1b-jhY21L693GmOfW?usp=sharing)
- **Source**: [M3DGR Dataset](https://github.com/sjtuyinjie/M3DGR)
- **Sensors**: Livox Avia / Mid-360 LiDAR + Built-in IMU

**Running Single Sequence**:
```bash
cd build

# Livox Avia
./lio_player ../config/avia.yaml /path/to/M3DGR/Dynamic03/avia

# Livox Mid-360
./lio_player ../config/mid360.yaml /path/to/M3DGR/Dynamic03/mid360
```

**Dataset Structure**:
```
M3DGR/
├── Dynamic03/
│   ├── avia/
│   │   ├── imu_data.csv
│   │   ├── lidar_timestamps.txt
│   │   └── lidar/
│   │       ├── 0000000000.pcd
│   │       ├── 0000000001.pcd
│   │       └── ...
│   └── mid360/
│       └── (same structure)
├── Dynamic04/
├── Occlusion03/
├── Occlusion04/
├── Outdoor01/
└── Outdoor04/
```


## Benchmark

Evaluation on [M3DGR Dataset](https://github.com/sjtuyinjie/M3DGR) comparing with FAST-LIO2.

### Livox Mid-360

| Sequence    | Ours (m) | FAST-LIO2 (m) | Ours (FPS) | FAST-LIO2 (FPS) |
|-------------|----------|---------------|------------|-----------------|
| Dynamic03   | 0.1456   | 0.1436        | 394        | 256             |
| Dynamic04   | 0.1549   | 0.1535        | 357        | 250             |
| Outdoor01   | 0.1536   | 0.1517        | 651        | 495             |
| Outdoor04   | 0.1562   | 0.1546        | 533        | 305             |
| Occlusion03 | 0.1269   | 0.1255        | 408        | 278             |
| Occlusion04 | 0.1410   | 0.1395        | 354        | 235             |
| **Average** | **0.1464** | **0.1447**  | **428**    | **285**         |

### Livox AVIA

| Sequence    | Ours (m) | FAST-LIO2 (m) | Ours (FPS) | FAST-LIO2 (FPS) |
|-------------|----------|---------------|------------|-----------------|
| Dynamic03   | 0.1478   | 0.1443        | 337        | 267             |
| Dynamic04   | 0.1563   | 0.1532        | 305        | 247             |
| Outdoor01   | 0.1549   | 0.1515        | 365        | 295             |
| Outdoor04   | 0.1560   | 0.1547        | 379        | 262             |
| Occlusion03 | 0.1252   | 0.1238        | 290        | 227             |
| Occlusion04 | 0.1402   | 0.1396        | 354        | 222             |
| **Average** | **0.1467** | **0.1445**  | **335**    | **251**         |

> **Note**: RPE RMSE (Relative Pose Error) is reported. Our method achieves **~1.5x faster** processing speed while maintaining comparable accuracy.


## Project Structure

```
lidar_inertial_odometry/
├── src/
│   ├── core/             # Core algorithm implementation
│   │   ├── Estimator.h/cpp                  # IEKF-based LIO estimator
│   │   ├── State.h/cpp                      # 18-dim state representation
│   │   ├── VoxelMap.h/cpp                   # Hash-based voxel map for fast KNN
│   │   └── ProbabilisticKernelOptimizer.h/cpp # PKO for adaptive robust estimation
│   │
│   ├── util/             # Utility functions
│   │   ├── LieUtils.h/cpp       # SO3/SE3 Lie group operations
│   │   ├── PointCloudUtils.h/cpp # Point cloud processing
│   │   └── ConfigUtils.h/cpp    # YAML configuration loader
│   │
│   └── viewer/           # Visualization
│       └── LIOViewer.h/cpp      # Pangolin-based 3D viewer
│
├── app/                  # Application executables
│   └── lio_player.cpp    # Dataset player with live visualization
│
├── config/               # Configuration files
│   ├── avia.yaml         # Parameters for Livox Avia LiDAR
│   └── mid360.yaml       # Parameters for Livox Mid-360 LiDAR
│
├── thirdparty/           # Third-party libraries
│   ├── pangolin/         # 3D visualization
│   └── spdlog/           # Logging (header-only)
│
├── CMakeLists.txt        # CMake build configuration
└── README.md             # This file
```



