#define PI 3.14159265358979323846

__global__ void update_positions(int* process_counts, int* grid, int* grid_sizes, float *positions, float *velocities, float *pressure, float *offsets, float box_size,
    const int max_grid_size, const float mass, const float dt, const int num_particles, const bool bounded, const bool offset_pos)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_particles) {
        atomicAdd(&process_counts[0], 1);
        positions[idx * 3] += velocities[idx * 3] * dt;
        positions[idx * 3 + 1] += velocities[idx * 3 + 1] * dt;
        positions[idx * 3 + 2] += velocities[idx * 3 + 2] * dt;
        if (offset_pos) {
            positions[idx * 3] += offsets[0];
            positions[idx * 3 + 1] += offsets[1];
            positions[idx * 3 + 2] += offsets[2];
        }
        if (bounded) {
            if (positions[idx * 3] > box_size) {
                atomicAdd(pressure, velocities[idx * 3]);
                positions[idx * 3] = 2 * box_size - positions[idx * 3];
                velocities[idx * 3] = -velocities[idx * 3];
            }
            if (positions[idx * 3] < 0) {
                atomicAdd(pressure, -velocities[idx * 3]);
                positions[idx * 3] = -positions[idx * 3];
                velocities[idx * 3] = -velocities[idx * 3];
            }
            if (positions[idx * 3 + 1] > box_size) {
                atomicAdd(pressure, velocities[idx * 3 + 1]);
                positions[idx * 3 + 1] = 2 * box_size - positions[idx * 3 + 1];
                velocities[idx * 3 + 1] = -velocities[idx * 3 + 1];
            }
            if (positions[idx * 3 + 1] < 0) {
                atomicAdd(pressure, -velocities[idx * 3 + 1]);
                positions[idx * 3 + 1] = -positions[idx * 3 + 1];
                velocities[idx * 3 + 1] = -velocities[idx * 3 + 1];
            }
            if (positions[idx * 3 + 2] > box_size) {
                atomicAdd(pressure, velocities[idx * 3 + 2]);
                positions[idx * 3 + 2] = 2 * box_size - positions[idx * 3 + 2];
                velocities[idx * 3 + 2] = -velocities[idx * 3 + 2];
            }
            if (positions[idx * 3 + 2] < 0) {
                atomicAdd(pressure, -velocities[idx * 3 + 2]);
                positions[idx * 3 + 2] = -positions[idx * 3 + 2];
                velocities[idx * 3 + 2] = -velocities[idx * 3 + 2];
            }
        } else {
            if (positions[idx * 3] > box_size) {
                atomicAdd(pressure, velocities[idx * 3]);
                positions[idx * 3] = fmod(positions[idx * 3], box_size);
            }
            if (positions[idx * 3] < 0) {
                atomicAdd(pressure, -velocities[idx * 3]);
                positions[idx * 3] = fmod(positions[idx * 3] + box_size, box_size);
            }
            if (positions[idx * 3 + 1] > box_size) {
                atomicAdd(pressure, velocities[idx * 3 + 1]);
                positions[idx * 3 + 1] = fmod(positions[idx * 3 + 1], box_size);
            }
            if (positions[idx * 3 + 1] < 0) {
                atomicAdd(pressure, -velocities[idx * 3 + 1]);
                positions[idx * 3 + 1] = fmod(positions[idx * 3 + 1] + box_size, box_size);
            }
            if (positions[idx * 3 + 2] > box_size) {
                atomicAdd(pressure, velocities[idx * 3 + 2]);
                positions[idx * 3 + 2] = fmod(positions[idx * 3 + 2], box_size);
            }
            if (positions[idx * 3 + 2] < 0) {
                atomicAdd(pressure, -velocities[idx * 3 + 2]);
                positions[idx * 3 + 2] = fmod(positions[idx * 3 + 2] + box_size, box_size);
            }
        }
        int i = int(positions[idx * 3]);
        int j = int(positions[idx * 3 + 1]);
        int k = int(positions[idx * 3 + 2]);
        i = 0 <= i && i < box_size ? i : i < 0 ? 0 : box_size - 1;
        j = 0 <= j && j < box_size ? j : j < 0 ? 0 : box_size - 1;
        k = 0 <= k && k < box_size ? k : k < 0 ? 0 : box_size - 1;
        int grid_idx = i * box_size * box_size + j * box_size + k;
        int grid_size = atomicAdd(&grid_sizes[grid_idx], 1);
        grid[grid_idx * max_grid_size + grid_size] = idx;
    }
}

__global__ void cell_states(int* process_counts, int* grid, int* grid_sizes, float* positions, float* velocities, float* us, float *p_tensor, const int box_size, const int max_grid_size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < box_size * box_size * box_size) {
        atomicAdd(&process_counts[2], 1);
        float ux = 0, uy = 0, uz = 0, u2 = 0, pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
        for (int i = 0; i < grid_sizes[idx]; i++) {
            int j = grid[idx * max_grid_size + i];
            ux += velocities[j * 3];
            uy += velocities[j * 3 + 1];
            uz += velocities[j * 3 + 2];
            u2 += velocities[j * 3] * velocities[j * 3] + velocities[j * 3 + 1] * velocities[j * 3 + 1] + velocities[j * 3 + 2] * velocities[j * 3 + 2];
        }
        ux /= grid_sizes[idx];
        uy /= grid_sizes[idx];
        uz /= grid_sizes[idx];
        u2 /= grid_sizes[idx];
        us[idx * 4] = ux;
        us[idx * 4 + 1] = uy;
        us[idx * 4 + 2] = uz;
        us[idx * 4 + 3] = u2;
        for (int i = 0; i < grid_sizes[idx]; i++) {
            int j = grid[idx * max_grid_size + i];
            pxx += (velocities[j * 3] - ux) * (velocities[j * 3] - ux);
            pyy += (velocities[j * 3 + 1] - uy) * (velocities[j * 3 + 1] - uy);
            pzz += (velocities[j * 3 + 2] - uz) * (velocities[j * 3 + 2] - uz);
            pxy += (velocities[j * 3] - ux) * (velocities[j * 3 + 1] - uy);
            pxz += (velocities[j * 3] - ux) * (velocities[j * 3 + 2] - uz);
            pyz += (velocities[j * 3 + 1] - uy) * (velocities[j * 3 + 2] - uz);
        }
        // pxx /= grid_sizes[idx];
        // pyy /= grid_sizes[idx];
        // pzz /= grid_sizes[idx];
        // pxy /= grid_sizes[idx];
        // pxz /= grid_sizes[idx];
        // pyz /= grid_sizes[idx];
        p_tensor[idx * 6] = pxx;
        p_tensor[idx * 6 + 1] = pyy;
        p_tensor[idx * 6 + 2] = pzz;
        p_tensor[idx * 6 + 3] = pxy;
        p_tensor[idx * 6 + 4] = pxz;
        p_tensor[idx * 6 + 5] = pyz;
    }
}

__device__ void scat22_o1(int idx, int jdx, int* grid, int* grid_sizes, float* positions, float* velocities, float* collision_rates,
                          const bool* j_offset,  // {j_offset_x, j_offset_y, j_offset_z}
                          const float* collision_rands,
                          const int max_grid_size,
                          const float dt,
                          const float d,
                          const int box_size,
                          const int direction
    ) {
    int i, j;
    float dx, dy, dz, dvx, dvy, dvz, dspeed, dist, dot;
    float vix_new, viy_new, viz_new, vjx_new, vjy_new, vjz_new;
    float dvx_new, dvy_new, dvz_new, kx, ky, kz, kist, collision_prob, k_factor;

    for (int i0 = 0; i0 < grid_sizes[idx]; i0++) {
        for (int j0 = 0; j0 < grid_sizes[jdx]; j0++) {
            i = grid[idx * max_grid_size + i0];
            j = grid[jdx * max_grid_size + j0];
            dx = positions[i * 3] - (j_offset[0] ? positions[j * 3] - box_size : positions[j * 3]);
            dy = positions[i * 3 + 1] - (j_offset[1] ? positions[j * 3 + 1] - box_size : positions[j * 3 + 1]);
            dz = positions[i * 3 + 2] - (j_offset[2] ? positions[j * 3 + 2] - box_size : positions[j * 3 + 2]);
            dvx = velocities[i * 3] - velocities[j * 3];
            dvy = velocities[i * 3 + 1] - velocities[j * 3 + 1];
            dvz = velocities[i * 3 + 2] - velocities[j * 3 + 2];
            dspeed = sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
            dist = sqrt(dx * dx + dy * dy + dz * dz);
            dx /= dist;
            dy /= dist;
            dz /= dist;
            dot = dx * dvx + dy * dvy + dz * dvz;
            vix_new = velocities[i * 3] - dot * dx;
            viy_new = velocities[i * 3 + 1] - dot * dy;
            viz_new = velocities[i * 3 + 2] - dot * dz;
            vjx_new = velocities[j * 3] + dot * dx;
            vjy_new = velocities[j * 3 + 1] + dot * dy;
            vjz_new = velocities[j * 3 + 2] + dot * dz;
            dvx_new = vix_new - vjx_new;
            dvy_new = viy_new - vjy_new;
            dvz_new = viz_new - vjz_new;
            kx = dvx - dvx_new;
            ky = dvy - dvy_new;
            kz = dvz - dvz_new;
            kist = sqrt(kx * kx + ky * ky + kz * kz);
            kx /= kist;
            ky /= kist;
            kz /= kist;
            switch (direction) {
                case -1:
                    k_factor = kx > 0 ? kx : 0;
                    break;
                case -2:
                    k_factor = ky > 0 ? ky : 0;
                    break;
                case -3:
                    k_factor = kz > 0 ? kz : 0;
                    break;
                case 1:
                    k_factor = kx < 0 ? -kx : 0;
                    break;
                case 2:
                    k_factor = ky < 0 ? -ky : 0;
                    break;
                case 3:
                    k_factor = kz < 0 ? -kz : 0;
                    break;
            } 
            collision_prob = dspeed * dt * PI * d * d * d / 2 * k_factor / test_particle;
            atomicAdd(&collision_rates[idx], collision_prob / grid_sizes[idx] / grid_sizes[jdx] / 3);
            if (collision_rands[idx * max_grid_size * max_grid_size + i0 * grid_sizes[idx] + j0] < collision_prob) {                            
                velocities[i * 3] = vix_new;
                velocities[i * 3 + 1] = viy_new;
                velocities[i * 3 + 2] = viz_new;
                velocities[j * 3] = vjx_new;
                velocities[j * 3 + 1] = vjy_new;
                velocities[j * 3 + 2] = vjz_new;
            }
        }
    }
}

__device__ void scat22_o2(int idx, int jdx, int* grid, int* grid_sizes, float* positions, float* velocities, float* collision_rates,
                          const bool* j_offset,  // {j_offset_x, j_offset_y, j_offset_z}
                          const float* collision_rands,
                          const int max_grid_size,
                          const float dt,
                          const float d,
                          const int box_size,
                          const int direction0,
                          const int direction1) {
    int i, j;
    float dx, dy, dz, dvx, dvy, dvz, dspeed, dist, dot;
    float vix_new, viy_new, viz_new, vjx_new, vjy_new, vjz_new;
    float dvx_new, dvy_new, dvz_new, kx, ky, kz, kist, collision_prob, k_factor;

    for (int i0 = 0; i0 < grid_sizes[idx]; i0++) {
        for (int j0 = 0; j0 < grid_sizes[jdx]; j0++) {
            i = grid[idx * max_grid_size + i0];
            j = grid[jdx * max_grid_size + j0];
            dx = positions[i * 3] - (j_offset[0] ? positions[j * 3] - box_size : positions[j * 3]);
            dy = positions[i * 3 + 1] - (j_offset[1] ? positions[j * 3 + 1] - box_size : positions[j * 3 + 1]);
            dz = positions[i * 3 + 2] - (j_offset[2] ? positions[j * 3 + 2] - box_size : positions[j * 3 + 2]);
            dvx = velocities[i * 3] - velocities[j * 3];
            dvy = velocities[i * 3 + 1] - velocities[j * 3 + 1];
            dvz = velocities[i * 3 + 2] - velocities[j * 3 + 2];
            dspeed = sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
            dist = sqrt(dx * dx + dy * dy + dz * dz);
            dx /= dist;
            dy /= dist;
            dz /= dist;
            dot = dx * dvx + dy * dvy + dz * dvz;
            vix_new = velocities[i * 3] - dot * dx;
            viy_new = velocities[i * 3 + 1] - dot * dy;
            viz_new = velocities[i * 3 + 2] - dot * dz;
            vjx_new = velocities[j * 3] + dot * dx;
            vjy_new = velocities[j * 3 + 1] + dot * dy;
            vjz_new = velocities[j * 3 + 2] + dot * dz;
            dvx_new = vix_new - vjx_new;
            dvy_new = viy_new - vjy_new;
            dvz_new = viz_new - vjz_new;
            kx = dvx - dvx_new;
            ky = dvy - dvy_new;
            kz = dvz - dvz_new;
            kist = sqrt(kx * kx + ky * ky + kz * kz);
            kx /= kist;
            ky /= kist;
            kz /= kist;
            switch (direction0) {
                case -1:
                    k_factor = kx > 0 ? kx : 0;
                    break;
                case -2:
                    k_factor = ky > 0 ? ky : 0;
                    break;
                case -3:
                    k_factor = kz > 0 ? kz : 0;
                    break;
                case 1:
                    k_factor = kx < 0 ? -kx : 0;
                    break;
                case 2:
                    k_factor = ky < 0 ? -ky : 0;
                    break;
                case 3:
                    k_factor = kz < 0 ? -kz : 0;
                    break;
            }
            switch (direction1) {
                case -1:
                    k_factor *= kx > 0 ? kx : 0;
                    break;
                case -2:
                    k_factor *= ky > 0 ? ky : 0;
                    break;
                case -3:
                    k_factor *= kz > 0 ? kz : 0;
                    break;
                case 1:
                    k_factor *= kx < 0 ? -kx : 0;
                    break;
                case 2:
                    k_factor *= ky < 0 ? -ky : 0;
                    break;
                case 3:
                    k_factor *= kz < 0 ? -kz : 0;
                    break;
            }
            collision_prob = dspeed * dt * PI * d * d * d * d / 8 * k_factor / test_particle;
            atomicAdd(&collision_rates[idx], collision_prob / grid_sizes[idx] / grid_sizes[jdx] / 6);
            if (collision_rands[idx * max_grid_size * max_grid_size + i0 * grid_sizes[idx] + j0] < collision_prob) {                            
                velocities[i * 3] = vix_new;
                velocities[i * 3 + 1] = viy_new;
                velocities[i * 3 + 2] = viz_new;
                velocities[j * 3] = vjx_new;
                velocities[j * 3 + 1] = vjy_new;
                velocities[j * 3 + 2] = vjz_new;
            }
        }
    }
}

__global__ void handle_collisions(int* process_counts, int* grid, int* grid_sizes, float* positions, float* velocities, float* collision_rates_o0, float* collision_rates_o1, float* collision_rates_o2,
                   const float d, const int box_size, const int max_grid_size, const float mass, const float dt, const int num_particles, const float* collision_rands, const int order, const bool bounded) {
    int i, j;
    float dx, dy, dz, dist, dvx, dvy, dvz, dot, dspeed, collision_prob;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < box_size * box_size * box_size) {
        atomicAdd(&process_counts[1], 1);
        for (int i0 = 0; i0 < grid_sizes[idx]; i0++) {
            for (int j0 = i0 + 1; j0 < grid_sizes[idx]; j0++) {
                i = grid[idx * max_grid_size + i0];
                j = grid[idx * max_grid_size + j0];
                dx = positions[i * 3] - positions[j * 3];
                dy = positions[i * 3 + 1] - positions[j * 3 + 1];
                dz = positions[i * 3 + 2] - positions[j * 3 + 2];
                dvx = velocities[i * 3] - velocities[j * 3];
                dvy = velocities[i * 3 + 1] - velocities[j * 3 + 1];
                dvz = velocities[i * 3 + 2] - velocities[j * 3 + 2];
                dspeed = sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
                collision_prob = dspeed * dt * PI * d * d / test_particle;
                atomicAdd(&collision_rates_o0[idx], collision_prob / grid_sizes[idx] / (grid_sizes[idx] - 1) * 2);
                if (collision_rands[idx * max_grid_size * max_grid_size + i0 * grid_sizes[idx] + j0] < collision_prob) {
                    dist = sqrt(dx * dx + dy * dy + dz * dz);
                    dx /= dist;
                    dy /= dist;
                    dz /= dist;
                    dot = dx * dvx + dy * dvy + dz * dvz;
                    velocities[i * 3] -= dot * dx;
                    velocities[i * 3 + 1] -= dot * dy;
                    velocities[i * 3 + 2] -= dot * dz;
                    velocities[j * 3] += dot * dx;
                    velocities[j * 3 + 1] += dot * dy;
                    velocities[j * 3 + 2] += dot * dz;
                }
            }
        }
        if (order >= 1) {
            int ibx, iby, ibz, jdx, jbx, jby, jbz;
            bool j_offset[3];
            ibx = idx / box_size / box_size;
            iby = idx / box_size % box_size;
            ibz = idx % box_size;
            jbx = ibx > 0 ? ibx - 1 : box_size - 1;
            jby = iby > 0 ? iby - 1 : box_size - 1;
            jbz = ibz > 0 ? ibz - 1 : box_size - 1;
            if (bounded) {
                j_offset[0] = false;
                j_offset[1] = false;
                j_offset[2] = false;
                if (ibx > 0) {
                    jdx = jbx * box_size * box_size + iby * box_size + ibz;
                    scat22_o1(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o1, j_offset, collision_rands, max_grid_size, dt, d, box_size, 1);
                }
                if (iby > 0) {
                    jdx = ibx * box_size * box_size + jby * box_size + ibz;
                    scat22_o1(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o1, j_offset, collision_rands, max_grid_size, dt, d, box_size, 2);
                }
                if (ibz > 0) {
                    jdx = ibx * box_size * box_size + iby * box_size + jbz;
                    scat22_o1(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o1, j_offset, collision_rands, max_grid_size, dt, d, box_size, 3);
                }
            } else {
                jdx = jbx * box_size * box_size + iby * box_size + ibz;
                j_offset[0] = ibx == 0;
                j_offset[1] = false;
                j_offset[2] = false;
                scat22_o1(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o1, j_offset, collision_rands, max_grid_size, dt, d, box_size, 1);
                jdx = ibx * box_size * box_size + jby * box_size + ibz;
                j_offset[0] = false;
                j_offset[1] = iby == 0;
                j_offset[2] = false;
                scat22_o1(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o1, j_offset, collision_rands, max_grid_size, dt, d, box_size, 2);
                j_offset[0] = false;
                j_offset[1] = false;
                j_offset[2] = ibz == 0;
                scat22_o1(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o1, j_offset, collision_rands, max_grid_size, dt, d, box_size, 3);
            }
            if (order >= 2) {
                int jbx2, jby2, jbz2;
                jbx2 = jbx > 0 ? jbx - 1 : box_size - 1;
                jby2 = jby > 0 ? jby - 1 : box_size - 1;
                jbz2 = jbz > 0 ? jbz - 1 : box_size - 1;
                if (bounded) {
                    j_offset[0] = false;
                    j_offset[1] = false;
                    j_offset[2] = false;
                    if (ibx > 1) {
                        jdx = jbx2 * box_size * box_size + iby * box_size + ibz;
                        scat22_o2(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o2, j_offset, collision_rands, max_grid_size, dt, d, box_size, 1, 1);
                    }
                    if (iby > 1) {
                        jdx = ibx * box_size * box_size + jby2 * box_size + ibz;
                        scat22_o2(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o2, j_offset, collision_rands, max_grid_size, dt, d, box_size, 2, 2);
                    }
                    if (ibz > 1) {
                        jdx = ibx * box_size * box_size + iby * box_size + jbz2;
                        scat22_o2(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o2, j_offset, collision_rands, max_grid_size, dt, d, box_size, 3, 3);
                    }
                    if (ibx > 0 && iby > 0) {
                        jdx = jbx * box_size * box_size + jby * box_size + ibz;
                        scat22_o2(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o2, j_offset, collision_rands, max_grid_size, dt, d, box_size, 1, 2);
                    }
                    if (ibx > 0 && ibz > 0) {
                        jdx = jbx * box_size * box_size + iby * box_size + jbz;
                        scat22_o2(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o2, j_offset, collision_rands, max_grid_size, dt, d, box_size, 1, 3);
                    }
                    if (iby > 0 && ibz > 0) {
                        jdx = ibx * box_size * box_size + jby * box_size + jbz;
                        scat22_o2(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o2, j_offset, collision_rands, max_grid_size, dt, d, box_size, 2, 3);
                    }
                } else {
                    jdx = jbx2 * box_size * box_size + iby * box_size + ibz;
                    j_offset[0] = ibx == 0 || jbx == 0;
                    j_offset[1] = false;
                    j_offset[2] = false;
                    scat22_o2(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o2, j_offset, collision_rands, max_grid_size, dt, d, box_size, 1, 1);
                    jdx = ibx * box_size * box_size + jby2 * box_size + ibz;
                    j_offset[0] = false;
                    j_offset[1] = iby == 0 || jby == 0;
                    j_offset[2] = false;
                    scat22_o2(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o2, j_offset, collision_rands, max_grid_size, dt, d, box_size, 2, 2);
                    jdx = ibx * box_size * box_size + iby * box_size + jbz2;
                    j_offset[0] = false;
                    j_offset[1] = false;
                    j_offset[2] = ibz == 0 || jbz == 0;
                    scat22_o2(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o2, j_offset, collision_rands, max_grid_size, dt, d, box_size, 3, 3);
                    jdx = jbx * box_size * box_size + jby * box_size + ibz;
                    j_offset[0] = ibx == 0;
                    j_offset[1] = iby == 0;
                    j_offset[2] = false;
                    scat22_o2(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o2, j_offset, collision_rands, max_grid_size, dt, d, box_size, 1, 2);
                    jdx = jbx * box_size * box_size + iby * box_size + jbz;
                    j_offset[0] = ibx == 0;
                    j_offset[1] = false;
                    j_offset[2] = ibz == 0;
                    scat22_o2(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o2, j_offset, collision_rands, max_grid_size, dt, d, box_size, 1, 3);
                    jdx = ibx * box_size * box_size + jby * box_size + jbz;
                    j_offset[0] = false;
                    j_offset[1] = iby == 0;
                    j_offset[2] = ibz == 0;
                    scat22_o2(idx, jdx, grid, grid_sizes, positions, velocities, collision_rates_o2, j_offset, collision_rands, max_grid_size, dt, d, box_size, 2, 3);
                }
            }
        }
    }
}