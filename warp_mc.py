import numpy as np
import warp as wp
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    f = wp.vec3(0.0, 0.0, 0.0)
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
                particle_v[index] += dx * dv_dx / d2
                particle_v[i] -= dx * dv_dx / d2
                break
                # f = f - dx * wp.max(dv_dx / d2, -V_MAX)
        elif d < dist_o1:
            if wp.randf(state) < - dv_dx / d * collision_prob_o1:
                particle_v[index] += dx * dv_dx / d2
                particle_v[i] -= dx * dv_dx / d2
                break
                # f = f - dx * wp.max(dv_dx / d2, -V_MAX)
        elif d < dist_o2:
            if wp.randf(state) < - dv_dx * dv_dx / d2 / dspeed * collision_prob_o2:
                particle_v[index] += dx * dv_dx / d2
                particle_v[i] -= dx * dv_dx / d2
                break
    particle_f[i] = f

@wp.kernel
def integrate_periodic(         # Apply periodic boundary conditions
    x: wp.array(dtype=wp.vec3), # pyright: ignore[reportInvalidTypeForm]
    v: wp.array(dtype=wp.vec3), # pyright: ignore[reportInvalidTypeForm]
    f: wp.array(dtype=wp.vec3), # pyright: ignore[reportInvalidTypeForm]
    p: wp.array(dtype=float),   # pyright: ignore[reportInvalidTypeForm]
    gravity: wp.vec3,
    dt: float,
    inv_mass: float,
    lower: wp.vec3,
    length: wp.vec3,
    offset: wp.vec3
):
    tid = wp.tid()
    # v[tid] += f[tid] * inv_mass * dt + gravity * dt
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
    gravity: wp.vec3,
    dt: float,
    inv_mass: float,
    lower: wp.vec3,
    length: wp.vec3
):
    tid = wp.tid()
    # v[tid] += f[tid] * inv_mass * dt + gravity * dt
    v[tid] += f[tid]
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

point_radius = 0.05
inv_mass = 1 / 200
temperature = 1.0
lower_boundary = wp.vec3(0.0, 0.0, 0.0)
L = 16.0
length = wp.vec3(L, L, L)

points = create_particle_volume(131072, lower_boundary, lower_boundary + length)
speeds = np.sqrt(temperature * inv_mass * 3)
thetas = np.random.rand(len(points)) * 2.0 * np.pi
phis = np.arccos(1 - 2 * np.random.rand(len(points)))
velocities = np.zeros((len(points), 3), dtype=float)
velocities[:, 0] = speeds * np.sin(phis) * np.cos(thetas)
velocities[:, 1] = speeds * np.sin(phis) * np.sin(thetas)
velocities[:, 2] = speeds * np.cos(phis)
speeds = wp.array(np.ones(len(points), dtype=float) * speeds, dtype=float)
velocities = wp.array(velocities, dtype=wp.vec3)

forces = wp.empty_like(points)
bounded_pressures = wp.zeros(len(points), dtype=float)
grid = wp.HashGrid(32, 32, 32)
grid_cell_size = 0.5
sim_dt = 0.005
sim_t = 100.0
# plot_pos(points)
pressure = np.zeros(int(sim_t / sim_dt), dtype=float)
pooled_pressures = []
mean_v2 = []
i = pre_i = 0
for t in tqdm(np.linspace(0, sim_t, int(sim_t / sim_dt))):
    grid.build(points, grid_cell_size)
    wp.launch(
            kernel=update_boltz,
            dim=points.shape,
            inputs=[
                grid.id,
                points,
                velocities,
                forces,
                grid_cell_size,
                grid_cell_size * 2,
                grid_cell_size * 3,
                grid_cell_size * 8,
                sim_dt,
                point_radius * 2,
                1
            ],
        )
    wp.launch(
            kernel=integrate_periodic,
            dim=points.shape,
            inputs=[
                points,
                velocities,
                forces,
                bounded_pressures,
                (0.0, 0.0, 0.0),
                sim_dt,
                inv_mass,
                lower_boundary,
                length,
                wp.vec3(wp.pi, wp.pi, 2.0)
            ],
        )
    pressure[i] = wp.to_torch(bounded_pressures).sum().item() / (length[0] * length[1] * 2 + length[1] * length[2] * 2 + length[2] * length[0] * 2)
    if t % 1.0 < sim_dt and i > 0:
        wp.launch(
            kernel=length_f,
            dim=points.shape,
            inputs=[
                velocities,
                speeds
            ],
        )
        # v2_torch = wp.to_torch(speeds) ** 2
        # print('\r', float(v2_torch.mean()), float(v2_torch.min()), float(v2_torch.max()), end='')
        mean_v2.append((wp.to_torch(speeds) ** 2).mean())
        pooled_pressures.append(pressure[pre_i:i+1].mean())
        pre_i = i
    i += 1
plot_pos(points, "output/positions.pdf")
plot_pos(velocities, "output/velocities.pdf")
hist, bins = np.histogram(speeds.numpy()**2, bins=200)
bin_widths = np.diff(bins)
bin_centers = (bins[:-1] + bins[1:]) / 2
norm_factor = bin_widths * np.sqrt(bin_centers)
normalized_hist = hist / norm_factor
plt.yscale('log')
plt.bar(bin_centers, normalized_hist, width=bin_widths, color='b', alpha=0.5, label='Data')
plt.savefig("output/dN_per_SqrtE_dE.pdf")
plt.close()

x_data = np.linspace(0, sim_t, len(pooled_pressures))
y_data = np.array(pooled_pressures)
params = np.polyfit(x_data, y_data, 6)
w = 30
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
plt.savefig("output/pressure.pdf")
plt.close()

y_data = np.array([v.cpu() for v in mean_v2])
fig, axs = plt.subplots(1, 2, figsize=(16, 6))
axs[0].scatter(x_data, y_data / inv_mass / 2.0, s=1, label='Data')
axs[0].legend()
axs[1].scatter(x_data, y_data / inv_mass / 2.0, s=1, label='Data')
axs[1].set_yscale('log')
axs[1].legend()
plt.tight_layout()
plt.legend()
plt.savefig("output/mean_energy.pdf")
plt.close()

plt.figure(figsize=(16, 12))
plt.hist(points.numpy()[:, 0], bins=256, histtype='step', alpha=0.5, label='X')
plt.hist(points.numpy()[:, 1], bins=256, histtype='step', alpha=0.5, label='Y')
plt.hist(points.numpy()[:, 2], bins=256, histtype='step', alpha=0.5, label='Z')
plt.grid()
plt.legend()
plt.savefig("output/position_histograms.pdf")
plt.close()

plt.figure(figsize=(32, 32))
plt.scatter(points.numpy()[:, 0], points.numpy()[:, 1], s=0.1, label='X-Y plane')
plt.scatter(points.numpy()[:, 0], points.numpy()[:, 2], s=0.1, label='X-Z plane')
plt.legend()
plt.savefig('points.pdf')
