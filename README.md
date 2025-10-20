# RT-RkNN

GPU Accelerated Reverse k-Nearest Neighbor Query using Ray Tracing Cores

## Overview

RT-RkNN is a high-performance implementation of reverse k-nearest neighbor (RkNN) queries leveraging NVIDIA GPU ray tracing cores for acceleration. This project explores various algorithmic approaches to efficiently solve RkNN queries on modern GPUs equipped with Ray Tracing (RT) cores, providing significant speedups over traditional CPU-based methods.

The reverse k-nearest neighbor query finds all points in a dataset that have a given query point among their k nearest neighbors. This is particularly useful in applications such as:
- Location-based services and spatial databases
- Data mining and pattern recognition
- Recommendation systems
- Network analysis

## Key Features

- **Hardware Acceleration**: Utilizes NVIDIA RT cores for ray tracing-based spatial queries
- **Multiple Algorithm Implementations**: Compare different approaches to RkNN queries
- **GPU Optimization**: Leverages CUDA for parallel processing

## Requirements

### System Requirements
- NVIDIA GPU with RT cores (RTX 20 series or newer recommended)
- Linux operating system (tested on Ubuntu)

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
make -j$(nproc)
```

The compiled executables will be available in the `build/` directory.

## Algorithm Implementations

The project includes several RkNN query implementations, each with different optimization strategies:

### 1. **rknn-rt** (Main Ray Tracing (RT) Approach)
- Primary ray tracing-based implementation
- Utilizes RT cores for accelerated spatial queries
- Optimized triangle mesh construction for query regions
- Best overall performance for most datasets

### 2. **rknn-tpl** (TPL Algorithm)
- Implementation of the Two-Phase List (TPL) algorithm
- Traditional approach for RkNN queries
- Useful for performance comparison baseline

### 3. **rknn-inf** (InfZone)
- Influence Zone-based RkNN algorithm
- Computes influence zones for efficient query processing
- Suitable for datasets with specific spatial distributions

### 4. **rknn-rt-all** (RT Approach with scene of ALL facilities)
- Creates triangle meshes on both far and near sides
- More comprehensive spatial coverage
- Higher accuracy at the cost of additional computation

### 5. **rknn-naive**
- Basic brute-force implementation
- Reference implementation for correctness verification
- Useful for small datasets and debugging

### 6. **rknn-rtree**
- R-tree based spatial indexing approach
- Classic data structure for spatial queries
- Good for datasets with hierarchical structure

## Usage

### Basic Usage

Run the main RT-based implementation:
```bash
./build/rknn-rt [options]
```

### Command Line Options

```bash
# Example usage with custom parameters
./build/rknn-rt -if dataset.txt -k 5 -q 10

# Run TPL algorithm
./build/rknn-tpl -if dataset.txt -k 5 -q 10
```

### Input Data Format

Data files should be in text format with points specified as:
```
id1 x1 y1
id2 x2 y2
...
```

Query files follow the same format for query points.


## Benchmarking

Run the evaluation script to compare different implementations:
```bash
cd build
./eva.sh
```

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