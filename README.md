# RT-RkNN

Accelerating Reverse k Nearest Neighbor Queries using Ray Tracing Cores

## Overview

RT-RkNN is a high-performance implementation of reverse k-nearest neighbor (RkNN) queries leveraging GPU ray tracing cores for acceleration. This project explores various algorithmic approaches to efficiently solve RkNN queries on modern GPUs equipped with Ray Tracing (RT) cores, providing significant speedups over traditional CPU and GPU-based methods.

The reverse k-nearest neighbor query finds all points in a dataset that have a given query point among their k nearest neighbors. This is particularly useful in applications such as:
- Location-based services and spatial databases
- Data mining and pattern recognition
- Recommendation systems
- Network analysis

## Key Features

- **Hardware Acceleration**: Utilizes RT cores for ray tracing-based spatial queries
- **Multiple Algorithm Implementations**: 9 different approaches including GPU-accelerated variants
- **Influence Zone Pruning**: Advanced geometric pruning using perpendicular bisectors
- **R\*-tree Spatial Indexing**: Efficient spatial data structures for query optimization

## Requirements

### System Requirements
- NVIDIA GPU with RT cores (RTX 20 series or newer recommended)
- Linux operating system (tested on Ubuntu 22.04.5 LTS)

### Software Dependencies
- **C++11** or higher
- **NVIDIA CUDA Toolkit 12.4**
- **NVIDIA OptiX 7.7**
- **CMake 3.10** or higher
- **GCC/G++** compatible with CUDA 12.4

### Additional Libraries (included in support/)
- GLFW (window management)
- GLAD (OpenGL loader)
- ImGui (user interface)

## Installation

### 1. Install Prerequisites

#### CUDA Toolkit 12.4
Download and install from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

```bash
# Verify CUDA installation
nvcc --version
```

#### OptiX 7.7
1. Download OptiX 7.7 SDK from [NVIDIA Developer OptiX](https://developer.nvidia.com/designworks/optix/download)
2. Extract and set the OptiX path in your environment:
```bash
export OptiX_INSTALL_DIR=/path/to/optix-7.7
```

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/RT-RkNN.git
cd RT-RkNN
```

### 3. Build the Project

```bash
mkdir -p build
cd build
cmake ..
make
```

The compiled executables will be available in the `build/bin` directory.

## Algorithm Implementations

The project includes **9 RkNN query implementations**, each with different optimization strategies:

### 1. **rknn-rt** (Main Ray Tracing Approach) - MOST OPTIMIZED
- Primary ray tracing-based implementation
- Utilizes NVIDIA OptiX RT cores
- Optimized triangle mesh construction
- Best overall performance for most datasets

### 2. **rknn-rt-direct** (Ray Tracing w/o meshes selection)
- Simplified ray tracing approach
- Utilizes NVIDIA OptiX RT cores
- Direct scene construction without complex preprocessing

### 3. **rknn-inf** (Influence Zone)
- CPU-based Influence Zone algorithm
- Perpendicular bisector-based pruning
- R*-tree traversal with line-based spatial filtering
- Useful for performance comparison baseline

### 4. **rknn-inf-gpu** (Influence Zone - GPU Accelerated)
- GPU-accelerated influence zone computation
- CUDA kernels for parallel bisector validation
- Device memory-optimized bisector checking

### 7. **rknn-tpl** (TPL)
- Implementation of the classic TPL algorithm
- The first half-space pruning method
- Perpendicular bisector computation with R*-tree
- Useful for performance comparison baseline

### 8. **rknn-slice** (SLICE)
- Divides space into 12 angular partitions (30° each)
- The state-of-the-art method using region-based pruning
- Useful for performance comparison baseline

### 9. **rknn-rtree** (Pure R*-tree Approach)
- Pure R*-tree based spatial indexing
- Classic data structure for spatial queries
- Simpler implementation for baseline comparison

### 5. **rknn-naive** (Brute Force)
- Basic CPU brute-force implementation
- Reference implementation for correctness verification
- Useful for small datasets and debugging

### 6. **rknn-naive-gpu** (Brute Force - GPU Accelerated)
- GPU-accelerated brute force baseline
- Parallel distance computation using CUDA
- k-nearest neighbor finding on GPU

## Usage

### Basic Usage

Run different algorithm implementations:

```bash
# Main ray tracing implementation (recommended)
./build/rknn-rt [options]

# GPU-accelerated influence zone (newest)
./build/rknn-inf-gpu [options]

# GPU-accelerated brute force baseline
./build/rknn-naive-gpu [options]

# Direct ray tracing variant
./build/rknn-rt-direct [options]

# Angular partition algorithm
./build/rknn-slice [options]

# Traditional algorithms
./build/rknn-tpl [options]
./build/rknn-inf [options]
./build/rknn-rtree [options]
./build/rknn-naive [options]
```

### Command Line Options

```bash
# Example usage with custom parameters
./build/rknn-rt -if dataset.txt -k 5 -q 10

# GPU-accelerated variants
./build/rknn-inf-gpu -if dataset.txt -k 5 -q 10
./build/rknn-naive-gpu -if dataset.txt -k 5 -q 10
```

### Input Data Format

Data files should be in text format with points specified as:
```
id1 x1 y1
id2 x2 y2
...
```

Query files follow the same format for query points.


Results will be saved in the `build/log/` directory.

## Troubleshooting

### Common Issues

1. **CUDA/OptiX not found during cmake**
   - Ensure environment variables are set correctly:
   ```bash
   export CUDA_HOME=/usr/local/cuda-12.4
   export OptiX_INSTALL_DIR=/path/to/optix-7.7
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
   ```

2. **Runtime errors with RT cores**
   - Verify your GPU supports RT cores: `nvidia-smi`
   - Update NVIDIA drivers to latest version

3. **Build failures**
   - Clean build directory: `rm -rf build/* && cd build && cmake .. && make`
   - Check GCC compatibility with CUDA 12.4


<!-- ## Citation

If you use this work in your research, please cite:
```bibtex
@software{rtrknn2024,
  title = {RT-RkNN: GPU-Accelerated Reverse k-Nearest Neighbor Query using Ray Tracing Cores},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/RT-RkNN}
}
``` -->

## License

This project is based on NVIDIA OptiX samples and follows the NVIDIA Software License Agreement. See the license headers in source files for details.

## Contact

For questions and support, please open an issue on GitHub or contact zhengyang.bai@riken.jp.