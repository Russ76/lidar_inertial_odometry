# LiDAR-Inertial Odometry

Tightly-coupled LIO using Iterated Extended Kalman Filter with 18-dimensional state space.

## Features

- 18-dim state: rotation, position, velocity, IMU biases, gravity
- Iterated Kalman Filter for LiDAR-IMU fusion
- Real-time 3D visualization (Pangolin)
- Custom implementations (no PCL/Sophus dependencies)

## Project Structure

```
lidar_inertial_odometry/
├── include/              # Header files
│   ├── Estimator.h       # Main LIO estimator
│   ├── State.h           # 18-dim state representation
│   ├── LieUtils.h        # SO3/SE3 Lie group utilities
│   ├── PointCloudUtils.h # Point cloud and KdTree
│   └── LIOViewer.h       # Pangolin-based viewer
│
├── src/                  # Implementation files
│   ├── Estimator.cpp
│   ├── State.cpp
│   ├── LieUtils.cpp
│   ├── PointCloudUtils.cpp
│   └── LIOViewer.cpp
│
├── app/                  # Application executables
│   └── lio_player.cpp    # Dataset player with visualization
│
├── CMakeLists.txt        # CMake build configuration
└── README.md             # This file
```


## Dependencies

### Required
- **CMake** (≥ 3.10)
- **Eigen3** (≥ 3.3): Linear algebra
- **Pangolin** (≥ 0.6): 3D visualization
- **GLEW**: OpenGL extension wrangling
- **spdlog** (≥ 1.8): Logging
