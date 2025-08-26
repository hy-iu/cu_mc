#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
// #include <vector>
// #include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "mc.h"

// CUDA 内核函数：初始化粒子速度和位置
__global__ void initialize_particles(float *rs, float *vs, float *speeds,
                                     float *thetas, float *phis,
                                     int n_particles, float len_grid,
                                     float temperature, float mass) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_particles)
    return;

  curandState state;
  curand_init(clock64(), idx, 0, &state);
  rs[idx * 3 + 0] = len_grid * curand_uniform(&state);
  rs[idx * 3 + 1] = len_grid * curand_uniform(&state);
  rs[idx * 3 + 2] = len_grid * curand_uniform(&state);

  speeds[idx] = sqrtf(temperature * 3 / mass);
  thetas[idx] = acosf(1 - 2 * curand_uniform(&state));
  phis[idx] = 2 * M_PI * curand_uniform(&state);

  vs[idx * 3 + 0] = speeds[idx] * sinf(thetas[idx]) * cosf(phis[idx]);
  vs[idx * 3 + 1] = speeds[idx] * sinf(thetas[idx]) * sinf(phis[idx]);
  vs[idx * 3 + 2] = speeds[idx] * cosf(thetas[idx]);
}

// CUDA 内核函数：更新粒子位置并处理边界条件
__global__ void update_positions(float *rs, float *vs, int n_particles,
                                 float dt, float len_grid) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_particles)
    return;

  // 更新位置
  rs[idx * 3 + 0] += vs[idx * 3 + 0] * dt;
  rs[idx * 3 + 1] += vs[idx * 3 + 1] * dt;
  rs[idx * 3 + 2] += vs[idx * 3 + 2] * dt;

  // 处理边界条件
  for (int i = 0; i < 3; i++) {
    if (rs[idx * 3 + i] < 0) {
      rs[idx * 3 + i] += LEN_PER_SIDE;
      // vs[idx * 3 + i] = -vs[idx * 3 + i];
    }
    if (rs[idx * 3 + i] > LEN_PER_SIDE) {
      rs[idx * 3 + i] -= LEN_PER_SIDE;
      // vs[idx * 3 + i] = -vs[idx * 3 + i];
    }
  }
}

// CUDA 内核函数：计算粒子所属的网格
__global__ void compute_grid_indices(int *flat_indices, float *rs,
                                     int n_particles, int n_grid_per_side,
                                     float len_grid) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_particles)
    return;

  int x = (int)(rs[idx * 3 + 0] / len_grid);
  int y = (int)(rs[idx * 3 + 1] / len_grid);
  int z = (int)(rs[idx * 3 + 2] / len_grid);

  x = min(x, n_grid_per_side - 1);
  y = min(y, n_grid_per_side - 1);
  z = min(z, n_grid_per_side - 1);

  flat_indices[idx] =
      x * n_grid_per_side * n_grid_per_side + y * n_grid_per_side + z;
}

__global__ void cl(thrust::device_vector<int> *grid_particle_list) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N_GRID_PER_SIDE * N_GRID_PER_SIDE * N_GRID_PER_SIDE)
    return;
  grid_particle_list[idx].clear();
}

__global__ void group_particles_by_grid(
    int *grid_particle_counts, thrust::device_vector<int> *grid_particle_list,
    const int *flat_indices, const int n_particles, const int n_grids) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_particles)
    return;

  int grid_idx = flat_indices[idx];

  int insert_idx = atomicAdd(&grid_particle_counts[grid_idx], 1);
  if (insert_idx < MAX_PARTICLES_PER_GRID) {
    grid_particle_list[grid_idx].push_back(idx);
  } else {
    printf("Grid %d is full!\n", grid_idx);
  }
}

// 主函数
int main() {
  // 参数设置
  const int n_particles = 16384;
  const float temperature = 0.4;
  const float mass = 200.0;
  const float dt = 0.1;
  // const float d = 0.1;
  const float stop = 100.0;
  const float len_grid = 1.0;
  const int n_grid_per_side = 8;

  // 分配主机和设备内存
  thrust::host_vector<float> h_rs(n_particles * 3);
  thrust::host_vector<float> h_vs(n_particles * 3);
  thrust::host_vector<float> h_speeds(n_particles);
  thrust::host_vector<float> h_thetas(n_particles);
  thrust::host_vector<float> h_phis(n_particles);
  thrust::host_vector<int> h_flat_indices(n_particles);
  thrust::host_vector<int> h_grid_particle_counts(
      n_grid_per_side * n_grid_per_side * n_grid_per_side);

  thrust::device_vector<float> d_rs = h_rs;
  thrust::device_vector<float> d_vs = h_vs;
  thrust::device_vector<float> d_speeds = h_speeds;
  thrust::device_vector<float> d_thetas = h_thetas;
  thrust::device_vector<float> d_phis = h_phis;
  thrust::device_vector<int> d_flat_indices = h_flat_indices;
  thrust::device_vector<int> d_grid_particle_counts = h_grid_particle_counts;
  thrust::device_vector<int>
      d_grid_particle_list[n_grid_per_side * n_grid_per_side * n_grid_per_side];

  // 初始化粒子
  int blocks = (n_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
  initialize_particles<<<blocks, BLOCK_SIZE>>>(
      thrust::raw_pointer_cast(d_rs.data()),
      thrust::raw_pointer_cast(d_vs.data()),
      thrust::raw_pointer_cast(d_speeds.data()),
      thrust::raw_pointer_cast(d_thetas.data()),
      thrust::raw_pointer_cast(d_phis.data()), n_particles, len_grid,
      temperature, mass);

  // 主循环
  for (int step = 0; step < stop / dt; step++) {
    std::cout << "\rStep: " << step;

    update_positions<<<blocks, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(d_rs.data()),
        thrust::raw_pointer_cast(d_vs.data()), n_particles, dt, len_grid);

    compute_grid_indices<<<blocks, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(d_flat_indices.data()),
        thrust::raw_pointer_cast(d_rs.data()), n_particles, n_grid_per_side,
        len_grid);
    thrust::fill(d_grid_particle_counts.begin(), d_grid_particle_counts.end(),
                 0);
    cl<<<(n_grid_per_side * n_grid_per_side * n_grid_per_side + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_grid_particle_list);
    group_particles_by_grid<<<blocks, BLOCK_SIZE>>>(
        thrust::raw_pointer_cast(d_grid_particle_counts.data()),
        d_grid_particle_list, thrust::raw_pointer_cast(d_flat_indices.data()),
        n_particles, n_grid_per_side);
  }

  // 将结果拷回主机
  h_rs = d_rs;
  h_vs = d_vs;

  // 输出结果
  std::cout << "\nSimulation complete!" << std::endl;

  return 0;
}