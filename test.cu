#define PI 3.14159265358979323846
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
// #include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/inner_product.h>

#include "mc.h"

#include <fstream>
#include <iostream>
struct printf_functor
{
  __host__ __device__
  void operator()(int x)
  {
    printf("%d\n", x);
  }
};
int main() {
    thrust::device_vector<float> vec(3);
    vec[0] = 0; vec[1] = 1; vec[2] = 2;

    // thrust::for_each(thrust::device, vec.begin(), vec.end(), printf_functor());
    thrust::device_vector<float> sums(5);
    sums[0] = thrust::reduce(thrust::device, vec.begin(), vec.end(), 0.0f);
    std::cout << sums[0] << std::endl;


    const int num_grids = 131072;
    float *collision_rates_o0;
    float *collision_rate_o0_s;
    cudaMalloc(&collision_rates_o0, sizeof(float) * num_grids);
    thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(collision_rates_o0);
    thrust::fill(thrust::device, dev_ptr, dev_ptr + num_grids, 1.0f);
    cudaMalloc(&collision_rate_o0_s, sizeof(float) * 50000);
    cudaMemset(collision_rate_o0_s, 0, sizeof(float) * 50000);
    cudaDeviceSynchronize();
    thrust::device_ptr<float> dev_ptr1 = thrust::device_pointer_cast(collision_rate_o0_s);

    cudaDeviceSynchronize();
    thrust::device_ptr<float> dev_ptr0 = thrust::device_pointer_cast(collision_rates_o0);
    dev_ptr1[0] = thrust::reduce(thrust::device, dev_ptr0, dev_ptr0 + num_grids);
    std::cout << dev_ptr1[0] << std::endl;
}
