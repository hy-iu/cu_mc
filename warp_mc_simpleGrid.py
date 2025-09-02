import numpy as np
import warp as wp
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import os
import sys
import shutil

wp.init()

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
def sample_unit_sphere_surface(output: wp.array(dtype=wp.vec3d)):
    tid = wp.tid()
    state = wp.rand_init(0, tid)
    output[tid] = wp.vec3d(wp.sample_unit_sphere_surface(state))

V_MAX = 1e5
@wp.kernel
def update_boltz(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3), # pyright: ignore[reportInvalidTypeForm]
    particle_v: wp.array(dtype=wp.vec3), # pyright: ignore[reportInvalidTypeForm]
    particle_f: wp.array(dtype=wp.vec3), # pyright: ignore[reportInvalidTypeForm]
    dist_o0: float,
    dist_o1: float,
    dist_o2: float,
    dist_potential_cut: float,
    dt: float,
    diameter: float,
    n_test: int
):
    tid = wp.tid()
    state = wp.rand_init(42, tid)
    i = wp.hash_grid_point_id(grid, tid)
    x = particle_x[i]
    v = particle_v[i]
    neighbors = wp.hash_grid_query(grid, x, dist_o2)
    if n_test == 1:
        collision_prob_o0 = dt * diameter * diameter * wp.pi / pow3(dist_o0)
        collision_prob_o1 = dt * diameter * diameter * diameter * wp.pi / 2.0 / pow4(dist_o0)
        collision_prob_o2 = dt * diameter * diameter * diameter * diameter * wp.pi / 8.0 / pow5(dist_o0)
    else:
        collision_prob_o0 = dt * diameter * diameter * wp.pi / float(n_test) / pow3(dist_o0)
        collision_prob_o1 = dt * diameter * diameter * diameter * wp.pi / float(n_test) / 2.0 / pow4(dist_o0)
        collision_prob_o2 = dt * diameter * diameter * diameter * diameter * wp.pi / float(n_test) / 8.0 / pow5(dist_o0)
    for index in neighbors:
        if index == i:
            continue
        dx = x - particle_x[index]
        dv = v - particle_v[index]
        d = wp.length(dx)
        d2 = wp.length_sq(dx)
        dspeed = wp.length(dv)
        dv_dx = wp.dot(dv, dx)
        if d < dist_o0:
            if wp.randf(state) < collision_prob_o0 * dspeed and dv_dx < 0:
                wp.atomic_add(particle_v, index, dx * dv_dx / d2)
                wp.atomic_add(particle_v, i, - dx * dv_dx / d2)
                break
        elif d < dist_o1:
            if wp.randf(state) < - dv_dx / d * collision_prob_o1:
                wp.atomic_add(particle_v, index, dx * dv_dx / d2)
                wp.atomic_add(particle_v, i, - dx * dv_dx / d2)
                break
        elif d < dist_o2:
            if wp.randf(state) < dv_dx * dv_dx / d2 / dspeed * collision_prob_o2 and dv_dx < 0:
                wp.atomic_add(particle_v, index, dx * dv_dx / d2)
                wp.atomic_add(particle_v, i, - dx * dv_dx / d2)
                break
    neighbors = wp.hash_grid_query(grid, x, dist_potential_cut)

@wp.kernel
def integrate_periodic(         # Apply periodic boundary conditions
    x: wp.array(dtype=wp.vec3), # pyright: ignore[reportInvalidTypeForm]
    v: wp.array(dtype=wp.vec3), # pyright: ignore[reportInvalidTypeForm]
    f: wp.array(dtype=wp.vec3), # pyright: ignore[reportInvalidTypeForm]
    p: wp.array(dtype=float),   # pyright: ignore[reportInvalidTypeForm]
    gravity: wp.vec3,           # not used
    dt: float,
    inv_mass: float,
    lower: wp.vec3,
    length: wp.vec3,
    offset: wp.vec3
):
    tid = wp.tid()
    v[tid] += f[tid] * inv_mass * dt
    v[tid] += f[tid]
    x[tid] += v[tid] * dt
    p[tid] = 0.0
    dx = x[tid] - lower
    for i in range(3):
        if dx[i] > length[i]:
            x[tid][i] = lower[i] + wp.mod(dx[i], length[i])
            p[tid] += 2.0 / (inv_mass * dt) * v[tid][i]
        elif dx[i] < 0.0:
            x[tid][i] = lower[i] + length[i] - wp.mod(-dx[i], length[i])
            p[tid] -= 2.0 / (inv_mass * dt) * v[tid][i]
    x[tid] = lower + wp.mod(x[tid] + offset - lower, length)

@wp.kernel
def integrate_bounce(
    x: wp.array(dtype=wp.vec3), # pyright: ignore[reportInvalidTypeForm]
    v: wp.array(dtype=wp.vec3), # pyright: ignore[reportInvalidTypeForm]
    f: wp.array(dtype=wp.vec3), # pyright: ignore[reportInvalidTypeForm]
    p: wp.array(dtype=float),   # pyright: ignore[reportInvalidTypeForm]
    gravity: wp.vec3,           # not used
    dt: float,
    inv_mass: float,
    lower: wp.vec3,
    length: wp.vec3
):
    tid = wp.tid()
    v[tid] += f[tid] * inv_mass * dt
    # v[tid] += f[tid]
    x[tid] += v[tid] * dt
    p[tid] = 0.0
    for i in range(3):
        while True:
            dx = x[tid] - lower
            if dx[i] > length[i]:
                x[tid][i] = lower[i] + 2.0 * length[i] - dx[i]
                p[tid] += 2.0 / (inv_mass * dt) * v[tid][i]
                v[tid][i] = -v[tid][i]
            elif dx[i] < 0.0:
                x[tid][i] = lower[i] - dx[i]
                p[tid] -= 2.0 / (inv_mass * dt) * v[tid][i]
                v[tid][i] = -v[tid][i]
            else:
                break

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


class ParticleSystemConfig:
    def __init__(self, N=131072, r=0.05, m=200, T=1.0, L=16.0, L_GRID=32, bx=-8.0, by=-8.0, bz=-8.0):
        self.num_particles = N
        self.point_radius = r
        self.mass = m
        self.inv_mass = 1 / m
        self.temperature = T
        self.L = L
        self.L_GRID = L_GRID
        self.bx = bx
        self.by = by
        self.bz = bz
        self.lower_boundary = wp.vec3(bx, by, bz)
        self.length = wp.vec3(self.L, self.L, self.L)

    def __str__(self):
        return json.dumps(self.__dict__, indent=2, default=lambda x: list(x) if isinstance(x, wp.vec3) else x)   


def load_particle_system_config(filename: str) -> ParticleSystemConfig:
    with open(filename, 'r') as f:
        data = json.load(f)
    assert {"num_particles", "point_radius", "mass", "temperature", "L", "L_GRID", "bx", "by", "bz"}.issubset(data.keys())
    return ParticleSystemConfig(
        N=data["num_particles"],
        r=data["point_radius"],
        m=data["mass"],
        T=data["temperature"],
        L=data["L"],
        L_GRID=data["L_GRID"],
        bx=data["bx"],
        by=data["by"],
        bz=data["bz"],
    )

def my_time_string(end: str = "") -> str:
    return str(datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S_%f_")) + f"{np.random.randint(0, 999):03d}" + end

class ParticleSystem:
    def __init__(self, c: ParticleSystemConfig):
        self.config = c
        self.points = create_particle_volume(c.num_particles, c.lower_boundary, c.lower_boundary + c.length)
        speeds = np.sqrt(c.temperature * c.inv_mass * 3)
        thetas = np.random.rand(len(self.points)) * 2.0 * np.pi
        phis = np.arccos(1 - 2 * np.random.rand(len(self.points)))
        velocities = np.zeros((len(self.points), 3), dtype=float)
        velocities[:, 0] = speeds * np.sin(phis) * np.cos(thetas)
        velocities[:, 1] = speeds * np.sin(phis) * np.sin(thetas)
        velocities[:, 2] = speeds * np.cos(phis)
        self.speeds = wp.array(np.ones(len(self.points), dtype=float) * speeds, dtype=float)
        self.velocities = wp.array(velocities, dtype=wp.vec3)

        self.forces = wp.empty_like(self.points)
        self.bounded_pressures = wp.zeros(len(self.points), dtype=float)

        self.num_grids: int = c.L_GRID ** 3
        self.max_grid_size: int = (c.num_particles // self.num_grids) * 4
        GridIndex = wp.vec(self.max_grid_size, dtype=wp.int32)
        self.grid = wp.array(shape=(c.L_GRID, c.L_GRID, c.L_GRID), dtype=GridIndex)
        self.grid_sizes = wp.zeros((c.L_GRID, c.L_GRID, c.L_GRID), dtype=wp.int32)
        # self.grid = wp.HashGrid(c.L_GRID, c.L_GRID, c.L_GRID)
        # self.grid_cell_size = c.L / c.L_GRID

    def evolve(self, output_dir=my_time_string(), sim_dt=0.005, sim_t=100.0, c=None):
        if c is None:
            c = self.config
        else:
            print(f"Warning: using provided config {c}")
        pressure = np.zeros(int(sim_t / sim_dt), dtype=float)
        pooled_pressures = []
        mean_v2 = []
        i = pre_i = 0
        for t in tqdm(np.linspace(0, sim_t, int(sim_t / sim_dt))):
            self.grid.build(self.points, self.grid_cell_size)
            wp.synchronize()
            wp.launch(
                    kernel=integrate_periodic,
                    dim=self.points.shape,
                    inputs=[
                        self.points,
                        self.velocities,
                        self.forces,
                        self.bounded_pressures,
                        (0.0, 0.0, 0.0),
                        sim_dt,
                        c.inv_mass,
                        c.lower_boundary,
                        c.length,
                        wp.vec3(wp.pi, wp.pi, 2.0)
                    ],
                )
            wp.synchronize()
            wp.launch(
                    kernel=update_boltz,
                    dim=self.points.shape,
                    inputs=[
                        self.grid.id,
                        self.points,
                        self.velocities,
                        self.forces,
                        self.grid_cell_size,
                        self.grid_cell_size * 2,
                        self.grid_cell_size * 3,
                        self.grid_cell_size * 8,
                        sim_dt,
                        c.point_radius * 2,
                        1
                    ],
                )
            wp.synchronize()
            pressure[i] = wp.to_torch(self.bounded_pressures).sum().item() / (c.length[0] * c.length[1] * 2 + c.length[1] * c.length[2] * 2 + c.length[2] * c.length[0] * 2)
            if t % 1.0 < sim_dt and i > 0:
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
                # self.velocities *= float(np.sqrt(c.temperature * c.inv_mass * 3 / mean_v2[-1].item()))
                pooled_pressures.append(pressure[pre_i:i+1].mean())
                pre_i = i
            i += 1
        os.makedirs(f"output/{output_dir}", exist_ok=False)
        shutil.copy(os.path.basename(__file__), f"output/{output_dir}/{os.path.basename(__file__)}")
        open(f"output/{output_dir}/config.json", 'w').write(str(c))
        np.save(f"output/{output_dir}/positions.npy", self.points.numpy())
        np.save(f"output/{output_dir}/velocities.npy", self.velocities.numpy())        
        # plot_pos(self.points, f"output/{output_dir}/positions.pdf")
        # plot_pos(self.velocities, f"output/{output_dir}/velocities.pdf")
        hist, bins = np.histogram(self.speeds.numpy()**2, bins=200)
        bin_widths = np.diff(bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        norm_factor = bin_widths * np.sqrt(bin_centers)
        normalized_hist = hist / norm_factor
        plt.yscale('log')
        plt.bar(bin_centers, normalized_hist, width=bin_widths, color='b', alpha=0.5, label='Data')
        plt.savefig(f"output/{output_dir}/dN_per_SqrtE_dE.pdf")
        plt.close()

        x_data = np.linspace(0, sim_t, len(pooled_pressures))
        y_data = np.array(pooled_pressures)
        np.save(f"output/{output_dir}/pressure.npy", y_data)
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
        plt.savefig(f"output/{output_dir}/pressure.pdf")
        plt.close()

        y_data = np.array([v.cpu() for v in mean_v2])
        np.save(f"output/{output_dir}/mean_v2.npy", y_data)
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        axs[0].scatter(x_data, y_data / c.inv_mass / 2.0, s=1, label='Data')
        axs[0].legend()
        axs[1].scatter(x_data, y_data / c.inv_mass / 2.0, s=1, label='Data')
        axs[1].set_yscale('log')
        axs[1].legend()
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"output/{output_dir}/mean_energy.pdf")
        plt.close()

        plt.figure(figsize=(16, 12))
        plt.hist(self.points.numpy()[:, 0], bins=256, histtype='step', alpha=0.5, label='X')
        plt.hist(self.points.numpy()[:, 1], bins=256, histtype='step', alpha=0.5, label='Y')
        plt.hist(self.points.numpy()[:, 2], bins=256, histtype='step', alpha=0.5, label='Z')
        plt.grid()
        plt.legend()
        plt.savefig(f"output/{output_dir}/position_histograms.pdf")
        plt.close()

        plt.figure(figsize=(16, 12))
        plt.hist(self.velocities.numpy()[:, 0], bins=256, histtype='step', alpha=0.5, label='VX')
        plt.hist(self.velocities.numpy()[:, 1], bins=256, histtype='step', alpha=0.5, label='VY')
        plt.hist(self.velocities.numpy()[:, 2], bins=256, histtype='step', alpha=0.5, label='VZ')
        plt.grid()
        plt.legend()
        plt.savefig(f"output/{output_dir}/velocity_histograms.pdf")
        plt.close()

        plt.figure(figsize=(32, 32))
        plt.scatter(self.points.numpy()[:, 0], self.points.numpy()[:, 1], s=0.1, label='X-Y plane')
        plt.scatter(self.points.numpy()[:, 0], self.points.numpy()[:, 2], s=0.1, label='X-Z plane')
        plt.legend()
        plt.savefig(f"output/{output_dir}/points.pdf")

if __name__ == "__main__":
    arg_r = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    print(f"Using point radius {arg_r}")
    c = ParticleSystemConfig(r=arg_r)
    ps = ParticleSystem(c)
    ps.evolve(output_dir=my_time_string('r'+str(arg_r)), sim_dt=0.005 * 0.05 / arg_r, sim_t=2.0 / arg_r)
