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

#define N_PARTICLES 16
#define N_GRID_PER_SIDE 4
#define BLOCK_SIZE 256

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

__global__ void compute_grid_indices(int *flat_indices, float *rs, float len_grid) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N_PARTICLES)
    return;
  int x = (int)(rs[idx * 3 + 0] / len_grid);
  int y = (int)(rs[idx * 3 + 1] / len_grid);
  int z = (int)(rs[idx * 3 + 2] / len_grid);

  x = min(x, N_GRID_PER_SIDE - 1);
  y = min(y, N_GRID_PER_SIDE - 1);
  z = min(z, N_GRID_PER_SIDE - 1);
  printf("x=%f, y=%f, z=%f, ix=%d, iy=%d, iz=%d\n", rs[idx * 3 + 0], rs[idx * 3 + 2], rs[idx * 3 + 2], x, y, z);
  flat_indices[idx] = x * N_GRID_PER_SIDE * N_GRID_PER_SIDE + y * N_GRID_PER_SIDE + z;
}

int main() {
  thrust::host_vector<float> h_rs(N_PARTICLES * 3);
  thrust::host_vector<float> h_vs(N_PARTICLES * 3);
  thrust::host_vector<float> h_speeds(N_PARTICLES);
  thrust::host_vector<float> h_thetas(N_PARTICLES);
  thrust::host_vector<float> h_phis(N_PARTICLES);
  thrust::host_vector<int> h_flat_indices(N_PARTICLES);
  thrust::host_vector<int> h_grid_particle_counts(N_GRID_PER_SIDE * N_GRID_PER_SIDE * N_GRID_PER_SIDE);

  thrust::device_vector<float> d_rs = h_rs;
  thrust::device_vector<float> d_vs = h_vs;
  thrust::device_vector<float> d_speeds = h_speeds;
  thrust::device_vector<float> d_thetas = h_thetas;
  thrust::device_vector<float> d_phis = h_phis;
  thrust::device_vector<int> d_flat_indices = h_flat_indices;
  thrust::device_vector<int> d_grid_particle_list[N_GRID_PER_SIDE * N_GRID_PER_SIDE * N_GRID_PER_SIDE];

  int blocks = (N_PARTICLES + BLOCK_SIZE - 1) / BLOCK_SIZE;
  initialize_particles<<<blocks, BLOCK_SIZE>>>(
      thrust::raw_pointer_cast(d_rs.data()), thrust::raw_pointer_cast(d_vs.data()),
      thrust::raw_pointer_cast(d_speeds.data()), thrust::raw_pointer_cast(d_thetas.data()),
      thrust::raw_pointer_cast(d_phis.data()), N_PARTICLES, N_GRID_PER_SIDE * 1.0f, 1.0, 200.0);
  for (int step = 0; step < 2; step++) {
    std::cout << "\rStep: " << step;

    compute_grid_indices<<<blocks, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_flat_indices.data()),
                                                 thrust::raw_pointer_cast(d_rs.data()), 1.0f);
  }
}