import numpy as np
import warp as wp
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import os
import sys
import shutil

N = 131072
RADIUS = 0.07
LJ_EPSILON = 0.1
MASS = 200
TEMPERATURE = 1.0
L = 16.0
L_GRID = 32
BX = -8.0
BY = -8.0
BZ = -8.0
N_TEST = 10

LOWER_BOUNDARY = wp.vec3(BX, BY, BZ)
LENGTH = wp.vec3(L, L, L)
GRID_CELL_SIZE = L / L_GRID
# OFFSET = wp.vec3(0.57, 0.57, 0.57)
OFFSET = wp.vec3(0.5, 0.5, 0.5)
NUM_GRIDS = L_GRID ** 3
MAX_GRID_SIZE = (N // NUM_GRIDS) * 4
# GridIndex = wp.vec(MAX_GRID_SIZE, dtype=wp.int32)

DT = 0.01
T_STOP = 100.0

V_MAX = 1e5  # haven't used
F_MAX = 1e5
D2_MIN = RADIUS * RADIUS * 0.1
PERIODOC_MAX = 5

OUTPUT_ROOT = 'warp_mc_simpleGrid_onlyLJ'

wp.init()

@wp.func
def lj_force(r2: wp.Float) -> wp.Float:
    inv_r2 = wp.static(4.0 * RADIUS * RADIUS) / r2
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r12 = inv_r6 * inv_r6
    f = 24.0 * LJ_EPSILON * (2.0 * inv_r12 - inv_r6) / r2
    return f if f < wp.static(F_MAX) else 0.0

@wp.kernel
def update_v_half_step(v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3)): # pyright: ignore[reportInvalidTypeForm]
    pid = wp.tid()
    v[pid] += f[pid] * wp.static(DT / (2.0 * MASS))

@wp.kernel
def update_periodic(grid: wp.array(dtype=wp.int32), grid_sizes: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3), p: wp.array(dtype=float)): # pyright: ignore[reportInvalidTypeForm]
    pid = wp.tid()
    x[pid] += v[pid] * DT
    p[pid] = 0.0
    for i in range(3):
        for _ in range(PERIODOC_MAX):
            dx = x[pid] - LOWER_BOUNDARY
            if dx[i] > LENGTH[i]:
                x[pid][i] = LOWER_BOUNDARY[i] + wp.mod(dx[i], LENGTH[i])
                p[pid] += wp.static(2.0 * MASS / DT) * v[pid][i]
            elif dx[i] < 0.0:
                x[pid][i] = LOWER_BOUNDARY[i] + LENGTH[i] - wp.mod(-dx[i], LENGTH[i])
                p[pid] -= wp.static(2.0 * MASS / DT) * v[pid][i]
            else:
                break
    x[pid] = LOWER_BOUNDARY + wp.mod(x[pid] + OFFSET - LOWER_BOUNDARY, LENGTH)
    i = wp.int(wp.floordiv(x[pid][0] - LOWER_BOUNDARY[0], GRID_CELL_SIZE))
    j = wp.int(wp.floordiv(x[pid][1] - LOWER_BOUNDARY[1], GRID_CELL_SIZE))
    k = wp.int(wp.floordiv(x[pid][2] - LOWER_BOUNDARY[2], GRID_CELL_SIZE))
    i = wp.min(wp.max(i, 0), L_GRID - 1)
    j = wp.min(wp.max(j, 0), L_GRID - 1)
    k = wp.min(wp.max(k, 0), L_GRID - 1)
    gid = i * L_GRID * L_GRID + j * L_GRID + k
    subid = wp.atomic_add(grid_sizes, gid, 1)
    grid[gid * MAX_GRID_SIZE + subid] = pid

@wp.kernel
def update_bounce(grid: wp.array(dtype=wp.int32), grid_sizes: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3), p: wp.array(dtype=float)): # pyright: ignore[reportInvalidTypeForm]
    pid = wp.tid()
    x[pid] += v[pid] * DT
    p[pid] = 0.0
    dx = x[pid] - LOWER_BOUNDARY
    for i in range(3):
        # while True:
        for _ in range(PERIODOC_MAX):
            dx = x[pid] - LOWER_BOUNDARY
            if dx[i] > LENGTH[i]:
                x[pid][i] = LOWER_BOUNDARY[i] + 2.0 * LENGTH[i] - dx[i]
                p[pid] += wp.static(2.0 * MASS / DT) * v[pid][i]
                v[pid][i] = -v[pid][i]
            elif dx[i] < 0.0:
                x[pid][i] = LOWER_BOUNDARY[i] - dx[i]
                p[pid] -= wp.static(2.0 * MASS / DT) * v[pid][i]
                v[pid][i] = -v[pid][i]
            else:
                break
    x[pid] = LOWER_BOUNDARY + wp.mod(x[pid] + OFFSET - LOWER_BOUNDARY, LENGTH)
    i = wp.int(wp.floordiv(x[pid][0] - LOWER_BOUNDARY[0], GRID_CELL_SIZE))
    j = wp.int(wp.floordiv(x[pid][1] - LOWER_BOUNDARY[1], GRID_CELL_SIZE))
    k = wp.int(wp.floordiv(x[pid][2] - LOWER_BOUNDARY[2], GRID_CELL_SIZE))
    i = wp.min(wp.max(i, 0), L_GRID - 1)
    j = wp.min(wp.max(j, 0), L_GRID - 1)
    k = wp.min(wp.max(k, 0), L_GRID - 1)
    gid = i * L_GRID * L_GRID + j * L_GRID + k
    subid = wp.atomic_add(grid_sizes, gid, 1)
    grid[gid * MAX_GRID_SIZE + subid] = pid

@wp.func
def scat22_o1(grid: wp.array(dtype=wp.int32), grid_sizes: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3), seed: int, ib: int, jb: int, direction: int, period_offset: wp.vec3): # pyright: ignore[reportInvalidTypeForm]
    state = wp.rand_init(seed)
    for i0 in range(grid_sizes[ib]):
        i = grid[ib * MAX_GRID_SIZE + i0]
        for j0 in range(grid_sizes[jb]):
            j = grid[jb * MAX_GRID_SIZE + j0]
            dx = x[i] - x[j] + period_offset
            d2 = wp.length_sq(dx)
            if d2 < wp.static(D2_MIN):  # too close, ignore
                continue
            if wp.randf(state) > 1 / N_TEST:
                continue
            # if d2 > wp.static(4.0 * RADIUS * RADIUS):
            f_lj = lj_force(d2) * dx
            wp.atomic_add(f, i, f_lj)
            wp.atomic_add(f, j, -f_lj)


@wp.func
def scat22_o2(grid: wp.array(dtype=wp.int32), grid_sizes: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3), seed: int, ib: int, jb: int, direction1: int, direction2: int, period_offset: wp.vec3): # pyright: ignore[reportInvalidTypeForm]
    state = wp.rand_init(seed)
    for i0 in range(grid_sizes[ib]):
        i = grid[ib * MAX_GRID_SIZE + i0]
        for j0 in range(grid_sizes[jb]):
            j = grid[jb * MAX_GRID_SIZE + j0]
            dx = x[i] - x[j] + period_offset
            d2 = wp.length_sq(dx)
            if d2 < wp.static(D2_MIN):  # too close, ignore
                continue
            if wp.randf(state) > 1 / N_TEST:
                continue
            # if d2 > wp.static(4.0 * RADIUS * RADIUS):
            f_lj = lj_force(d2) * dx
            wp.atomic_add(f, i, f_lj)
            wp.atomic_add(f, j, -f_lj)

@wp.kernel
def update_collisions(grid: wp.array(dtype=wp.int32), grid_sizes: wp.array(dtype=wp.int32), x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), f: wp.array(dtype=wp.vec3), seed: int): # pyright: ignore[reportInvalidTypeForm]
    gid = wp.tid()
    state = wp.rand_init(seed, gid)
    for i0 in range(grid_sizes[gid]):
        i = grid[gid * MAX_GRID_SIZE + i0]
        for j0 in range(i0 + 1, grid_sizes[gid]):
            j = grid[gid * MAX_GRID_SIZE + j0]
            dx = x[i] - x[j]
            d2 = wp.length_sq(dx)
            if d2 < wp.static(D2_MIN):  # too close, ignore
                continue
            if wp.randf(state) > 1 / N_TEST:
                continue
            # if d2 > wp.static(4.0 * RADIUS * RADIUS):
            f_lj = lj_force(d2) * dx
            wp.atomic_add(f, i, f_lj)
            wp.atomic_add(f, j, -f_lj)
    ibx = gid // (L_GRID * L_GRID)
    iby = gid // L_GRID % L_GRID
    ibz = gid % L_GRID
    jbx = ibx - 1 if ibx > 0 else L_GRID - 1
    jby = iby - 1 if iby > 0 else L_GRID - 1
    jbz = ibz - 1 if ibz > 0 else L_GRID - 1
    xoff = 0.0 if ibx > 0 else LENGTH[0]
    yoff = 0.0 if iby > 0 else LENGTH[1]
    zoff = 0.0 if ibz > 0 else LENGTH[2]
    scat22_o1(grid, grid_sizes, x, v, f, seed * 2 + gid, gid, jbx * L_GRID * L_GRID + iby * L_GRID + ibz, 0, wp.vec3(xoff, 0.0, 0.0))
    scat22_o1(grid, grid_sizes, x, v, f, seed * 3 + gid, gid, ibx * L_GRID * L_GRID + jby * L_GRID + ibz, 1, wp.vec3(0.0, yoff, 0.0))
    scat22_o1(grid, grid_sizes, x, v, f, seed * 4 + gid, gid, ibx * L_GRID * L_GRID + iby * L_GRID + jbz, 2, wp.vec3(0.0, 0.0, zoff))
    jbx2 = ibx - 2 if ibx > 1 else L_GRID - 1
    jby2 = iby - 2 if iby > 1 else L_GRID - 1
    jbz2 = ibz - 2 if ibz > 1 else L_GRID - 1
    xoff2 = 0.0 if ibx > 1 else LENGTH[0]
    yoff2 = 0.0 if iby > 1 else LENGTH[1]
    zoff2 = 0.0 if ibz > 1 else LENGTH[2]
    scat22_o2(grid, grid_sizes, x, v, f, seed * 5 + gid, gid, jbx * L_GRID * L_GRID + jby * L_GRID + ibz, 0, 1, wp.vec3(xoff, yoff, 0.0))
    scat22_o2(grid, grid_sizes, x, v, f, seed * 6 + gid, gid, jbx * L_GRID * L_GRID + iby * L_GRID + jbz, 0, 2, wp.vec3(xoff, 0.0, zoff))
    scat22_o2(grid, grid_sizes, x, v, f, seed * 7 + gid, gid, ibx * L_GRID * L_GRID + jby * L_GRID + jbz, 1, 2, wp.vec3(0.0, yoff, zoff))
    scat22_o2(grid, grid_sizes, x, v, f, seed * 8 + gid, gid, jbx2 * L_GRID * L_GRID + iby * L_GRID + ibz, 0, 0, wp.vec3(xoff2, 0.0, 0.0))
    scat22_o2(grid, grid_sizes, x, v, f, seed * 9 + gid, gid, ibx * L_GRID * L_GRID + jby2 * L_GRID + ibz, 1, 1, wp.vec3(0.0, yoff2, 0.0))
    scat22_o2(grid, grid_sizes, x, v, f, seed * 10 + gid, gid, ibx * L_GRID * L_GRID + iby * L_GRID + jbz2, 2, 2, wp.vec3(0.0, 0.0, zoff2))
    for i0 in range(grid_sizes[gid]):
        i = grid[gid * MAX_GRID_SIZE + i0]
        for j0 in range(grid_sizes[jbx * L_GRID * L_GRID + jby * L_GRID + jbz]):
            j = grid[(gid + 1 if gid + 1 < NUM_GRIDS else 0) * MAX_GRID_SIZE + j0]
            dx = x[i] - x[j] + wp.vec3(xoff, yoff, zoff)
            d2 = wp.length_sq(dx)
            # if d2 > wp.static(4.0 * RADIUS * RADIUS):
            f_lj = lj_force(d2) * dx
            wp.atomic_add(f, i, f_lj)
            wp.atomic_add(f, j, -f_lj)

@wp.func
def pow3(x: wp.Float) -> wp.Float:
    return x * x * x

@wp.func
def pow4(x: wp.Float) -> wp.Float:
    return x * x * x * x

@wp.func
def pow5(x: wp.Float) -> wp.Float:
    return x * x * x * x * x

@wp.kernel
def length_f(velocities: wp.array(dtype=wp.vec3), speeds: wp.array(dtype=float)): # pyright: ignore[reportInvalidTypeForm]
    speeds[wp.tid()] = wp.length(velocities[wp.tid()])

@wp.kernel
def sample_unit_sphere_surface(output: wp.array(dtype=wp.vec3)): # pyright: ignore[reportInvalidTypeForm]
    tid = wp.tid()
    state = wp.rand_init(0, tid)
    output[tid] = wp.vec3(wp.sample_unit_sphere_surface(state))


def create_particle_volume(
    num_particles: int,
    lower: wp.vec3,
    higher: wp.vec3,
) -> wp.array(dtype=wp.vec3): # pyright: ignore[reportInvalidTypeForm]
    v = higher - lower
    points = np.random.rand(num_particles, 3) * v + lower
    print(f"N={num_particles}, n={num_particles / v[0] / v[1] / v[2]:.2f}")
    return wp.array(points, dtype=wp.vec3)

def plot_pos(points, filename):
    plt.figure(figsize=(16, 12))
    plt.scatter(points.numpy()[:, 0], points.numpy()[:, 1], s=0.1, label='X-Y plane')
    plt.scatter(points.numpy()[:, 0], points.numpy()[:, 2], s=0.1, label='X-Z plane')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def my_time_string(end: str = "") -> str:
    return str(datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S_%f_")) + f"{np.random.randint(0, 999):03d}" + end

class ParticleSystem:
    def __init__(self):
        print(f"RADIUS={RADIUS}, epsilon={LJ_EPSILON}, L={L}, L_GRID={L_GRID}, GRID_CELL_SIZE={GRID_CELL_SIZE}, MAX_GRID_SIZE={MAX_GRID_SIZE}, NUM_GRIDS={NUM_GRIDS}")
        print(f"MASS={MASS}, T={TEMPERATURE}, dt={DT}, N_test={N_TEST}, t_stop={T_STOP}")
        self.points = create_particle_volume(N, LOWER_BOUNDARY, LOWER_BOUNDARY + LENGTH)
        self.velocities = wp.empty_like(self.points)
        wp.launch(
            kernel=sample_unit_sphere_surface,
            dim=self.points.shape,
            inputs=[self.velocities],
        )
        wp.synchronize()
        self.velocities *= float(np.sqrt(TEMPERATURE / MASS * 3))
        self.speeds = wp.zeros(N, dtype=float)
        wp.launch(kernel=length_f, dim=self.points.shape, inputs=[self.velocities, self.speeds])
        wp.synchronize()
        self.forces = wp.empty_like(self.points)
        self.bounded_pressures = wp.zeros(N, dtype=float)

        self.grid = wp.array(shape=NUM_GRIDS*MAX_GRID_SIZE, dtype=wp.int32)
        self.grid_sizes = wp.zeros(NUM_GRIDS, dtype=wp.int32)

    def evolve(self, output_dir=my_time_string()):
        pressure = np.zeros(int(T_STOP / DT), dtype=float)
        pooled_pressures = []
        mean_v2 = []
        i = pre_i = 0
        self.forces.zero_()
        for t in tqdm(np.linspace(0, T_STOP, int(T_STOP / DT))):
            self.grid_sizes.zero_()
            self.grid.zero_()
            wp.launch(
                kernel=update_v_half_step,
                dim=N,
                inputs=[
                    self.velocities,
                    self.forces,
                ],
            )
            wp.synchronize()
            wp.launch(
                    kernel=update_periodic,
                    dim=N,
                    inputs=[
                        self.grid,
                        self.grid_sizes,
                        self.points,
                        self.velocities,
                        self.forces,
                        self.bounded_pressures,
                    ],
                )
            wp.synchronize()
            # print(f"{i}: {t:.3f}s")
            self.forces.zero_()
            wp.launch(
                    kernel=update_collisions,
                    dim=NUM_GRIDS,
                    inputs=[
                        self.grid,
                        self.grid_sizes,
                        self.points,
                        self.velocities,
                        self.forces,
                        np.random.randint(0, 9999999),
                    ],
                )
            wp.synchronize()
            wp.launch(
                kernel=update_v_half_step,
                dim=N,
                inputs=[
                    self.velocities,
                    self.forces,
                ],
            )
            # print(f"{t:.3f}s")
            pressure[i] = wp.to_torch(self.bounded_pressures).sum().item() / (LENGTH[0] * LENGTH[1] * 2 + LENGTH[1] * LENGTH[2] * 2 + LENGTH[2] * LENGTH[0] * 2)
            if t % 0.1 < DT and i > 0:
                wp.launch(
                    kernel=length_f,
                    dim=self.points.shape,
                    inputs=[
                        self.velocities,
                        self.speeds
                    ],
                )
                # v2_torch = wp.to_torch(speeds) ** 2
                # print('\r', float(v2_torch.mean()), float(v2_torch.min()), float(v2_torch.max()), end='')
                mean_v2.append((wp.to_torch(self.speeds) ** 2).mean())
                # supply energy
                self.velocities *= float(np.sqrt(TEMPERATURE / MASS * 3 / mean_v2[-1].item()))
                pooled_pressures.append(pressure[pre_i:i+1].mean())
                pre_i = i
            i += 1
        os.makedirs(f"{OUTPUT_ROOT}/{output_dir}", exist_ok=False)
        shutil.copy(os.path.basename(__file__), f"{OUTPUT_ROOT}/{output_dir}/{os.path.basename(__file__)}")
        np.save(f"{OUTPUT_ROOT}/{output_dir}/positions.npy", self.points.numpy())
        np.save(f"{OUTPUT_ROOT}/{output_dir}/velocities.npy", self.velocities.numpy())        
        # plot_pos(self.points, f"{OUTPUT_ROOT}/{output_dir}/positions.pdf")
        # plot_pos(self.velocities, f"{OUTPUT_ROOT}/{output_dir}/velocities.pdf")
        hist, bins = np.histogram(self.speeds.numpy()**2, bins=200)
        bin_widths = np.diff(bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        norm_factor = bin_widths * np.sqrt(bin_centers)
        normalized_hist = hist / norm_factor
        plt.yscale('log')
        plt.bar(bin_centers, normalized_hist, width=bin_widths, color='b', alpha=0.5, label='Data')
        plt.savefig(f"{OUTPUT_ROOT}/{output_dir}/dN_per_SqrtE_dE.pdf")
        plt.close()

        x_data = np.linspace(0, T_STOP, len(pooled_pressures))
        y_data = np.array(pooled_pressures)
        np.save(f"{OUTPUT_ROOT}/{output_dir}/pressure.npy", y_data)
        # params = np.polyfit(x_data, y_data, 6)
        w = min(10, int(len(y_data) / 2))
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        # Linear scale
        axs[0].scatter(x_data, y_data, s=1, label='Data')
        axs[0].scatter(x_data[w-1:], np.convolve(y_data, np.ones(w), "valid") / w, s=1, label='Pooled Mean')
        axs[0].set_ylabel('pressure')
        axs[0].legend()
        # Log scale
        axs[1].scatter(x_data, y_data, s=1, label='Data')
        axs[1].scatter(x_data[w-1:], np.convolve(y_data, np.ones(w), "valid") / w, s=1, label='Pooled Mean')
        axs[1].set_yscale('log')
        axs[1].set_ylabel('pressure')
        axs[1].legend()
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{OUTPUT_ROOT}/{output_dir}/pressure.pdf")
        plt.close()

        y_data = np.array([v.cpu() for v in mean_v2])
        np.save(f"{OUTPUT_ROOT}/{output_dir}/mean_v2.npy", y_data)
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        axs[0].scatter(x_data, y_data * MASS / 2.0, s=1, label='Data')
        axs[0].legend()
        axs[1].scatter(x_data, y_data * MASS / 2.0, s=1, label='Data')
        axs[1].set_yscale('log')
        axs[1].legend()
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{OUTPUT_ROOT}/{output_dir}/mean_energy.pdf")
        plt.close()

        plt.figure(figsize=(16, 12))
        plt.hist(self.points.numpy()[:, 0], bins=256, histtype='step', alpha=0.5, label='X')
        plt.hist(self.points.numpy()[:, 1], bins=256, histtype='step', alpha=0.5, label='Y')
        plt.hist(self.points.numpy()[:, 2], bins=256, histtype='step', alpha=0.5, label='Z')
        plt.grid()
        plt.legend()
        plt.savefig(f"{OUTPUT_ROOT}/{output_dir}/position_histograms.pdf")
        plt.close()

        plt.figure(figsize=(16, 12))
        plt.hist(self.velocities.numpy()[:, 0], bins=256, histtype='step', alpha=0.5, label='VX')
        plt.hist(self.velocities.numpy()[:, 1], bins=256, histtype='step', alpha=0.5, label='VY')
        plt.hist(self.velocities.numpy()[:, 2], bins=256, histtype='step', alpha=0.5, label='VZ')
        plt.grid()
        plt.legend()
        plt.savefig(f"{OUTPUT_ROOT}/{output_dir}/velocity_histograms.pdf")
        plt.close()

        plt.figure(figsize=(32, 32))
        plt.scatter(self.points.numpy()[:, 0], self.points.numpy()[:, 1], s=0.1, label='X-Y plane')
        plt.scatter(self.points.numpy()[:, 0], self.points.numpy()[:, 2], s=0.1, label='X-Z plane')
        plt.legend()
        plt.savefig(f"{OUTPUT_ROOT}/{output_dir}/points.pdf")

if __name__ == "__main__":
    ps = ParticleSystem()
    ps.evolve(output_dir=my_time_string('r'+str(RADIUS)))
