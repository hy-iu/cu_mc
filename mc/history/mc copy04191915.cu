#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
// #include <vector>
// #include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "mc.h"

#include <fstream>
#include <iostream>

// CUDA 内核函数：初始化粒子速度和位置
__global__ void initialize_particles(float *rs,
                                     float *vs,
                                     float *speeds,
                                     float *thetas,
                                     float *phis,
                                     int n_particles,
                                     float len_grid,
                                     float temperature,
                                     float mass) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_particles)
    return;

  curandState state;
  curand_init(clock64(), idx, 0, &state);
  rs[idx * 3 + 0] = LEN_PER_SIDE * curand_uniform(&state);
  rs[idx * 3 + 1] = LEN_PER_SIDE * curand_uniform(&state);
  rs[idx * 3 + 2] = LEN_PER_SIDE * curand_uniform(&state);

  speeds[idx] = sqrtf(temperature * 3 / mass);
  thetas[idx] = acosf(1 - 2 * curand_uniform(&state));
  phis[idx] = 2 * M_PI * curand_uniform(&state);

  vs[idx * 3 + 0] = speeds[idx] * sinf(thetas[idx]) * cosf(phis[idx]);
  vs[idx * 3 + 1] = speeds[idx] * sinf(thetas[idx]) * sinf(phis[idx]);
  vs[idx * 3 + 2] = speeds[idx] * cosf(thetas[idx]);
}

// CUDA 内核函数：更新粒子位置并处理边界条件
__global__ void update_positions(float *rs, float *vs, float dt, float *pressure, float len_grid) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N_PARTICLES)
    return;

  // 更新位置
  rs[idx * 3 + 0] += vs[idx * 3 + 0] * dt;
  rs[idx * 3 + 1] += vs[idx * 3 + 1] * dt;
  rs[idx * 3 + 2] += vs[idx * 3 + 2] * dt;

  // 处理边界条件
  for (int i = 0; i < 3; i++) {
    if (rs[idx * 3 + i] < 0) {
      rs[idx * 3 + i] += LEN_PER_SIDE;
      atomicAdd(pressure, -2 * MASS * vs[idx * 3 + i] / 6 / LEN_PER_SIDE / LEN_PER_SIDE / dt);
      // vs[idx * 3 + i] = -vs[idx * 3 + i];
    }
    if (rs[idx * 3 + i] > LEN_PER_SIDE) {
      rs[idx * 3 + i] -= LEN_PER_SIDE;
      atomicAdd(pressure, 2 * MASS * vs[idx * 3 + i] / 6 / LEN_PER_SIDE / LEN_PER_SIDE / dt);
      // vs[idx * 3 + i] = -vs[idx * 3 + i];
    }
  }
}

// CUDA 内核函数：计算粒子所属的网格
__global__ void compute_grid_indices(int *flat_indices, float *rs, int *grid_particle_counts, float len_grid) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N_PARTICLES)
    return;
  int x = (int)(rs[idx * 3 + 0] / len_grid);
  int y = (int)(rs[idx * 3 + 1] / len_grid);
  int z = (int)(rs[idx * 3 + 2] / len_grid);

  x = min(x, N_GRID_PER_SIDE - 1);
  y = min(y, N_GRID_PER_SIDE - 1);
  z = min(z, N_GRID_PER_SIDE - 1);

  flat_indices[idx] = x * N_GRID_PER_SIDE * N_GRID_PER_SIDE + y * N_GRID_PER_SIDE + z;
}

__global__ void clear_grid_particle_counts(int *grid_particle_counts) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N_GRID_PER_SIDE * N_GRID_PER_SIDE * N_GRID_PER_SIDE) {
    grid_particle_counts[idx] = 0;
  }
}

__global__ void group_particles(int *grid_particle_counts, int *grid_particle_list, int *flat_indices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N_PARTICLES)
    return;

  int grid_idx = flat_indices[idx];

  // atomic 操作将粒子加入对应网格
  int insert_idx = atomicAdd(&grid_particle_counts[grid_idx], 1);
  if (insert_idx < MAX_PARTICLES_PER_GRID) {
    grid_particle_list[grid_idx * MAX_PARTICLES_PER_GRID + insert_idx] = idx;
  } else {
    printf("Grid %d is full! insert_idx=%d\n", grid_idx, insert_idx);
    cudaError(cudaGetLastError());
  }
}

__device__ unsigned int xorshift32(unsigned int &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

__global__ void
shuffle_particles_kernel(int *grid_particle_list, const int *grid_particle_counts, const int L, unsigned int seed) {
  // 三维线程索引对应网格坐标
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (i >= L || j >= L || k >= L)
    return;

  // 计算网格的线性索引
  int grid_idx = i + j * L + k * L * L;
  int count = grid_particle_counts[grid_idx];
  if (count <= 1)
    return;

  int *particles = &grid_particle_list[grid_idx * MAX_PARTICLES_PER_GRID];
  unsigned int state = (seed + grid_idx) * 123456789 + 1;
  // Fisher-Yates洗牌算法
  for (int idx = count - 1; idx > 0; --idx) {
    unsigned int rand_val = xorshift32(state) % (idx + 1);
    int temp = particles[idx];
    particles[idx] = particles[rand_val];
    particles[rand_val] = temp;
  }
}

void shuffle_particles(int *d_grid_particle_list, const int *d_grid_particle_counts, unsigned int seed) {
  const int threads_per_dim = THREADS_PER_DIM_SHUFFLING; // 每个维度的线程数
  const int L = N_GRID_PER_SIDE;
  dim3 threads(threads_per_dim, threads_per_dim, threads_per_dim);
  dim3 blocks((L + threads.x - 1) / threads.x, (L + threads.y - 1) / threads.y, (L + threads.z - 1) / threads.z);

  shuffle_particles_kernel<<<blocks, threads>>>(d_grid_particle_list, d_grid_particle_counts, L, seed);
  cudaDeviceSynchronize();
}

__global__ void scatt_o0(float *rs,
                         float *vs,
                         int *grid_counts,
                         int *grid_list,
                         float d,
                         float dt,
                         float temperature,
                         float mass,
                         int n_particles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int _g = idx / (MAX_PARTICLES_PER_GRID / 2); // 这样可能迫使MAX_PARTICLES_PER_GRID要设为偶数
  int _i = idx % (MAX_PARTICLES_PER_GRID / 2);
  int half_grid_size = grid_counts[_g] / 2;
  if (_g >= N_GRID_PER_SIDE * N_GRID_PER_SIDE * N_GRID_PER_SIDE) {
    printf("idx=%d\n", _g); // for debug
    return;
  }
  if (grid_counts[_g] <= 0) {
    if (grid_counts[_g] < 0) // for debug
      printf("grid_counts[%d]=%d\n", _g, grid_counts[_g]);
    return;
  }
  if (_i >= half_grid_size) // 这样可能过于浪费了
    return;
  int _p0 = grid_list[_g * MAX_PARTICLES_PER_GRID + _i];
  int _p1 = grid_list[_g * MAX_PARTICLES_PER_GRID + _i + half_grid_size];
  if (_p0 >= n_particles || _p1 >= n_particles) // for debug
    return;
  auto dx = rs[_p0 * 3 + 0] - rs[_p1 * 3 + 0];
  auto dy = rs[_p0 * 3 + 1] - rs[_p1 * 3 + 1];
  auto dz = rs[_p0 * 3 + 2] - rs[_p1 * 3 + 2];
  auto dr2 = dx * dx + dy * dy + dz * dz;
  auto dvx = vs[_p0 * 3 + 0] - vs[_p1 * 3 + 0];
  auto dvy = vs[_p0 * 3 + 1] - vs[_p1 * 3 + 1];
  auto dvz = vs[_p0 * 3 + 2] - vs[_p1 * 3 + 2];
  auto dv2 = dvx * dvx + dvy * dvy + dvz * dvz;
  auto dspeed = sqrtf(dv2);
  float collision_prob = dspeed * dt * M_PI * d * d / TEST_PARTICLE;
  curandState state;
  if (curand_uniform(&state) < collision_prob) {
    auto dist = sqrtf(dr2);
    dx /= dist;
    dy /= dist;
    dz /= dist;
    auto dot = dx * dvx + dy * dvy + dz * dvz;
    // vs[_p0 * 3 + 0] -= dx * dot * 2.0f;
    vs[_p0 * 3 + 0] -= dx * dot;
    vs[_p0 * 3 + 1] -= dy * dot;
    vs[_p0 * 3 + 2] -= dz * dot;
    vs[_p1 * 3 + 0] += dx * dot;
    vs[_p1 * 3 + 1] += dy * dot;
    vs[_p1 * 3 + 2] += dz * dot;
  }
}

__global__ void scatt_o1(float *rs,
                         float *vs,
                         int *grid_counts,
                         int *grid_list,
                         float d,
                         float dt,
                         float temperature,
                         float mass,
                         int n_particles,
                         int direction) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int _g0 = idx / (MAX_PARTICLES_PER_GRID / 2);
  int _i0 = idx % (MAX_PARTICLES_PER_GRID / 2);
  int half_grid_size = grid_counts[_g0] / 2;
  int _i1 = _i0 + half_grid_size;
  int _g1;
  auto rec_x = false, rec_y = false, rec_z = false;
  if (direction == 2) {
    _g1 = _g0 + 1;
    if (_g1 >= N_GRID_TOTAL) {
      rec_z = true;
      _g1 %= N_GRID_TOTAL;
      // return;
    }
  } else if (direction == 1) {
    _g1 = _g0 + N_GRID_PER_SIDE;
    if (_g1 >= N_GRID_TOTAL) {
      rec_y = true;
      _g1 %= N_GRID_TOTAL;
      // return;
    }
  } else if (direction == 0) {
    _g1 = _g0 + N_GRID_PER_SIDE * N_GRID_PER_SIDE;
    if (_g1 >= N_GRID_TOTAL) {
      rec_x = true;
      _g1 %= N_GRID_TOTAL;
      // return;
    }
  } else {
    // TODO: how to catch error
    return;
  }
  if (_i0 >= grid_counts[_g0] || _i1 >= grid_counts[_g1]) // 这样可能过于浪费了
    return;
  int _p0 = grid_list[_g0 * MAX_PARTICLES_PER_GRID + _i0];
  int _p1 = grid_list[_g1 * MAX_PARTICLES_PER_GRID + _i1];
  if (_p0 >= n_particles || _p1 >= n_particles) // for debug
    return;
  float dx = rs[_p0 * 3 + 0] - rs[_p1 * 3 + 0] - (rec_x ? LEN_PER_SIDE : 0.0f);
  float dy = rs[_p0 * 3 + 1] - rs[_p1 * 3 + 1] - (rec_y ? LEN_PER_SIDE : 0.0f);
  float dz = rs[_p0 * 3 + 2] - rs[_p1 * 3 + 2] - (rec_z ? LEN_PER_SIDE : 0.0f);
  float dr2 = dx * dx + dy * dy + dz * dz;
  float dvx = vs[_p0 * 3 + 0] - vs[_p1 * 3 + 0];
  float dvy = vs[_p0 * 3 + 1] - vs[_p1 * 3 + 1];
  float dvz = vs[_p0 * 3 + 2] - vs[_p1 * 3 + 2];
  float dv2 = dvx * dvx + dvy * dvy + dvz * dvz;
  float dspeed = sqrtf(dv2);
  float collision_prob = dspeed * dt * M_PI * d * d * d / 2.0f / TEST_PARTICLE;
  float dist = sqrtf(dr2);
  dx /= dist;
  dy /= dist;
  dz /= dist;
  float dot = dx * dvx + dy * dvy + dz * dvz;
  float v0x_new = vs[_p0 * 3 + 0] - dx * dot;
  float v0y_new = vs[_p0 * 3 + 1] - dy * dot;
  float v0z_new = vs[_p0 * 3 + 2] - dz * dot;
  float v1x_new = vs[_p1 * 3 + 0] + dx * dot;
  float v1y_new = vs[_p1 * 3 + 1] + dy * dot;
  float v1z_new = vs[_p1 * 3 + 2] + dz * dot;
  float dvx_new = v0x_new - v1x_new;
  float dvy_new = v0y_new - v1y_new;
  float dvz_new = v0z_new - v1z_new;
  float kx = dvx - dvx_new;
  float ky = dvy - dvy_new;
  float kz = dvz - dvz_new;
  float kist = sqrtf(kx * kx + ky * ky + kz * kz);
  kx /= kist;
  ky /= kist;
  kz /= kist;
  // if (direction == 0) {
  //   if (kx < 0)
  //     return;
  //   else
  //     collision_prob *= kx;
  // } else if (direction == 1) {
  //   if (ky < 0)
  //     return;
  //   else
  //     collision_prob *= ky;
  // } else if (direction == 2) {
  //   if (kz < 0)
  //     return;
  //   else
  //     collision_prob *= kz;
  // } else {
  //   // TODO: how to catch error
  //   return;
  // }
  if (dot < 0)
    return;
  curandState state;
  if (curand_uniform(&state) < collision_prob) {
    vs[_p0 * 3 + 0] = v0x_new;
    vs[_p0 * 3 + 1] = v0y_new;
    vs[_p0 * 3 + 2] = v0z_new;
    vs[_p1 * 3 + 0] = v1x_new;
    vs[_p1 * 3 + 1] = v1y_new;
    vs[_p1 * 3 + 2] = v1z_new;
  }
}

__global__ void scatt_o2(float *rs,
                         float *vs,
                         int *grid_counts,
                         int *grid_list,
                         float d,
                         float dt,
                         float temperature,
                         float mass,
                         int n_particles,
                         int direction0,
                         int direction1) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int _g0 = idx / (MAX_PARTICLES_PER_GRID / 2);
  int _i0 = idx % (MAX_PARTICLES_PER_GRID / 2);
  int half_grid_size = grid_counts[_g0] / 2;
  int _i1 = _i0 + half_grid_size;
  int _g1;
  auto rec_x = false, rec_y = false, rec_z = false;
  if (direction0 == 2) {
    _g1 = _g0 + 1;
    if (_g1 >= N_GRID_TOTAL) {
      rec_z = true;
      _g1 %= N_GRID_TOTAL;
      // return;
    }
  } else if (direction0 == 1) {
    _g1 = _g0 + N_GRID_PER_SIDE;
    if (_g1 >= N_GRID_TOTAL) {
      rec_y = true;
      _g1 %= N_GRID_TOTAL;
      // return;
    }
  } else if (direction0 == 0) {
    _g1 = _g0 + N_GRID_PER_SIDE * N_GRID_PER_SIDE;
    if (_g1 >= N_GRID_TOTAL) {
      rec_x = true;
      _g1 %= N_GRID_TOTAL;
      // return;
    }
  } else {
    // TODO: how to catch error
    return;
  }
  if (direction1 == 2) {
    _g1 = _g1 + 1;
    if (_g1 >= N_GRID_TOTAL) {
      rec_z = true;
      _g1 %= N_GRID_TOTAL;
      // return;
    }
  } else if (direction1 == 1) {
    _g1 = _g1 + N_GRID_PER_SIDE;
    if (_g1 >= N_GRID_TOTAL) {
      rec_y = true;
      _g1 %= N_GRID_TOTAL;
      // return;
    }
  } else if (direction1 == 0) {
    _g1 = _g1 + N_GRID_PER_SIDE * N_GRID_PER_SIDE;
    if (_g1 >= N_GRID_TOTAL) {
      rec_x = true;
      _g1 %= N_GRID_TOTAL;
      // return;
    }
  } else {
    // TODO: how to catch error
    return;
  }
  if (_i0 >= grid_counts[_g0] || _i1 >= grid_counts[_g1]) // 这样可能过于浪费了
    return;
  int _p0 = grid_list[_g0 * MAX_PARTICLES_PER_GRID + _i0];
  int _p1 = grid_list[_g1 * MAX_PARTICLES_PER_GRID + _i1];
  if (_p0 >= n_particles || _p1 >= n_particles) // for debug
    return;
  float dx = rs[_p0 * 3 + 0] - rs[_p1 * 3 + 0] - (rec_x ? LEN_PER_SIDE : 0.0f);
  float dy = rs[_p0 * 3 + 1] - rs[_p1 * 3 + 1] - (rec_y ? LEN_PER_SIDE : 0.0f);
  float dz = rs[_p0 * 3 + 2] - rs[_p1 * 3 + 2] - (rec_z ? LEN_PER_SIDE : 0.0f);
  float dr2 = dx * dx + dy * dy + dz * dz;
  float dvx = vs[_p0 * 3 + 0] - vs[_p1 * 3 + 0];
  float dvy = vs[_p0 * 3 + 1] - vs[_p1 * 3 + 1];
  float dvz = vs[_p0 * 3 + 2] - vs[_p1 * 3 + 2];
  float dv2 = dvx * dvx + dvy * dvy + dvz * dvz;
  float dspeed = sqrtf(dv2);
  float collision_prob = dspeed * dt * M_PI * d * d * d * d / 8.0f / TEST_PARTICLE;
  float dist = sqrtf(dr2);
  dx /= dist;
  dy /= dist;
  dz /= dist;
  float dot = dx * dvx + dy * dvy + dz * dvz;
  float v0x_new = vs[_p0 * 3 + 0] - dx * dot;
  float v0y_new = vs[_p0 * 3 + 1] - dy * dot;
  float v0z_new = vs[_p0 * 3 + 2] - dz * dot;
  float v1x_new = vs[_p1 * 3 + 0] + dx * dot;
  float v1y_new = vs[_p1 * 3 + 1] + dy * dot;
  float v1z_new = vs[_p1 * 3 + 2] + dz * dot;
  float dvx_new = v0x_new - v1x_new;
  float dvy_new = v0y_new - v1y_new;
  float dvz_new = v0z_new - v1z_new;
  float kx = dvx - dvx_new;
  float ky = dvy - dvy_new;
  float kz = dvz - dvz_new;
  float kist = sqrtf(kx * kx + ky * ky + kz * kz);
  kx /= kist;
  ky /= kist;
  kz /= kist;
  if (direction0 == 0) {
    if (kx < 0)
      return;
    else
      collision_prob *= kx;
  } else if (direction0 == 1) {
    if (ky < 0)
      return;
    else
      collision_prob *= ky;
  } else if (direction0 == 2) {
    if (kz < 0)
      return;
    else
      collision_prob *= kz;
  } else {
    // TODO: how to catch error
    return;
  }
  switch (direction1) {
  case 0:
    if (kx < 0)
      return;
    else
      collision_prob *= kx;
    break;
  case 1:
    if (ky < 0)
      return;
    else
      collision_prob *= ky;
    break;
  case 2:
    if (kz < 0)
      return;
    else
      collision_prob *= kz;
    break;
  default:
    assert(false);
  }
  curandState state;
  if (curand_uniform(&state) < collision_prob) {
    vs[_p0 * 3 + 0] = v0x_new;
    vs[_p0 * 3 + 1] = v0y_new;
    vs[_p0 * 3 + 2] = v0z_new;
    vs[_p1 * 3 + 0] = v1x_new;
    vs[_p1 * 3 + 1] = v1y_new;
    vs[_p1 * 3 + 2] = v1z_new;
  }
}

int main() {
  const int n_particles = N_PARTICLES;
  const float T = TEMPERATURE;
  const float mass = MASS;
  const float dt = DT;
  const float d = ENSKOG_D;
  const float stop = T_STOP;
  const float len_grid = LEN_GRID;
  const int n_grids = N_GRID_PER_SIDE * N_GRID_PER_SIDE * N_GRID_PER_SIDE;

  float *rs;
  float *vs;
  float *speeds;
  float *thetas;
  float *phis;
  int *flat_indices;
  int *grid_counts;
  int *grid_list;
  float *pressures;

  const int B = BLOCK_SIZE;
  const int G = (N_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cudaMalloc((void **)&rs, n_particles * 3 * sizeof(float));
  cudaMalloc((void **)&vs, n_particles * 3 * sizeof(float));
  cudaMalloc((void **)&speeds, n_particles * sizeof(float));
  cudaMalloc((void **)&thetas, n_particles * sizeof(float));
  cudaMalloc((void **)&phis, n_particles * sizeof(float));
  cudaMalloc((void **)&flat_indices, n_particles * sizeof(int));
  cudaMalloc((void **)&grid_counts, n_grids * sizeof(int));
  cudaMalloc((void **)&grid_list, n_grids * MAX_PARTICLES_PER_GRID * sizeof(int));
  cudaMalloc((void **)&pressures, int(stop / dt) * sizeof(float));

  initialize_particles<<<G, B>>>(rs, vs, speeds, thetas, phis, n_particles, len_grid, T, mass);
  // cudaDeviceSynchronize();

  // int *h_grid_list = new int[n_grids * MAX_PARTICLES_PER_GRID];
  // std::ofstream ls_file("ls.txt");
  for (int step = 0; step < stop / dt; step++) {
    float temp = 0.0f;
    cudaMemcpy(&pressures[step], &temp, sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "\rStep: " << step;
    update_positions<<<G, B>>>(rs, vs, dt * 10, &pressures[step], len_grid);
    cudaDeviceSynchronize();
    compute_grid_indices<<<G, B>>>(flat_indices, rs, grid_counts, len_grid);
    cudaDeviceSynchronize();
    clear_grid_particle_counts<<<(n_grids + B - 1) / B, B>>>(grid_counts);
    group_particles<<<G, B>>>(grid_counts, grid_list, flat_indices);

    shuffle_particles(grid_list, grid_counts, step);
    // cudaMemcpy(h_grid_list, grid_list, n_grids * MAX_PARTICLES_PER_GRID * sizeof(int), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < n_grids; i++) {
    //   for (int j = 0; j < MAX_PARTICLES_PER_GRID; j++) {
    //     ls_file << h_grid_list[i * MAX_PARTICLES_PER_GRID + j] << " ";
    //   }
    //   ls_file << "\n";
    // }
    scatt_o0<<<(n_grids * MAX_PARTICLES_PER_GRID / 2 + B - 1) / B, B>>>(rs, vs, grid_counts, grid_list, d, dt, T, mass,
                                                                        n_particles);
    cudaDeviceSynchronize();
    shuffle_particles(grid_list, grid_counts, step);
    scatt_o1<<<(n_grids * MAX_PARTICLES_PER_GRID / 2 + B - 1) / B, B>>>(rs, vs, grid_counts, grid_list, d, dt, T, mass,
                                                                        n_particles, 0);
    cudaDeviceSynchronize();
    shuffle_particles(grid_list, grid_counts, step);
    scatt_o1<<<(n_grids * MAX_PARTICLES_PER_GRID / 2 + B - 1) / B, B>>>(rs, vs, grid_counts, grid_list, d, dt, T, mass,
                                                                        n_particles, 1);
    cudaDeviceSynchronize();
    shuffle_particles(grid_list, grid_counts, step);
    scatt_o1<<<(n_grids * MAX_PARTICLES_PER_GRID / 2 + B - 1) / B, B>>>(rs, vs, grid_counts, grid_list, d, dt, T, mass,
                                                                        n_particles, 2);
    // for (int i0 = 0; i0 < 3; i0++) {
    //   for (int i1 = 0; i1 < 3; i1++) {
    //     cudaDeviceSynchronize();
    //     shuffle_particles(grid_list, grid_counts, step);
    //     scatt_o2<<<(n_grids * MAX_PARTICLES_PER_GRID / 2 + B - 1) / B, B>>>(rs, vs, grid_counts, grid_list, d, dt, T,
    //                                                                         mass, n_particles, i0, i1);
    //   }
    // }
  }
  // save rs and vs to file
  std::ofstream rs_file("rs.txt");
  std::ofstream vs_file("vs.txt");
  float *h_rs = (float *)malloc(n_particles * 3 * sizeof(float));
  float *h_vs = (float *)malloc(n_particles * 3 * sizeof(float));
  float *h_p = (float *)malloc(stop / dt * sizeof(float));
  cudaMemcpy(h_rs, rs, n_particles * 3 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_vs, vs, n_particles * 3 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_p, pressures, stop / dt * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n_particles; i++) {
    rs_file << h_rs[i * 3 + 0] << ", " << h_rs[i * 3 + 1] << ", " << h_rs[i * 3 + 2] << std::endl;
    vs_file << h_vs[i * 3 + 0] << ", " << h_vs[i * 3 + 1] << ", " << h_vs[i * 3 + 2] << std::endl;
  }
  rs_file.close();
  vs_file.close();
  // ls_file.close();
  std::ofstream pressure_file("pressure.txt");
  for (int i = 0; i < stop / dt; i++) {
    pressure_file << h_p[i] << std::endl;
  }
  pressure_file.close();
  free(h_rs);
  free(h_vs);

  std::cout << "\nSimulation complete!" << std::endl;

  return 0;
}