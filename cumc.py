import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
from pycuda.curandom import rand as curand
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


MOD_compute_velocities = """
//cuda
__global__ void compute_velocities(float *velocities, float *theta, float *phi, float E0, float mass, int num_particles)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num_particles) {
        velocities[idx * 3] = sqrt(2 * E0 / mass) * sin(phi[idx]) * cos(theta[idx]);
        velocities[idx * 3 + 1] = sqrt(2 * E0 / mass) * sin(phi[idx]) * sin(theta[idx]);
        velocities[idx * 3 + 2] = sqrt(2 * E0 / mass) * cos(phi[idx]);
    }
}
//!cuda
"""
with open("mc.cu", "r") as f:
    MOD_MC = f.read()

class Box:
    def __init__(self, box_size = 16, num_particles = 131072, stop = 100, dt = 0.01, dt_get_states = 1, T = 1, mass = 200, d = 0.1, bounded = False, n_test=1, order = 2, offset_pos=False, offsets=(0.2,0.2,0.2)) -> None:
        self.box_size = box_size
        self.num_particles = num_particles
        self.stop = stop
        self.dt = dt
        self.T = T
        self.mass = mass
        self.d = d
        self.bounded = bounded
        self.particle_positions = curand((num_particles, 3)) * box_size
        self.offsets = gpuarray.to_gpu(np.array(offsets, np.float32))
        self.E0 = 1.5 * T
        self.theta = curand(num_particles) * 2 * np.pi
        self.phi = gpuarray.zeros(num_particles, np.float32)
        self.phi.set(np.arccos(1 - 2 * curand(num_particles).get()))
        self.particle_velocities = gpuarray.zeros((num_particles, 3), np.float32)
        self.mod = SourceModule(MOD_compute_velocities)
        self.compute_velocities = self.mod.get_function("compute_velocities")
        self.compute_velocities(self.particle_velocities, self.theta, self.phi, np.float32(self.E0), np.float32(mass), np.int32(num_particles), block=(256,1,1), grid=(num_particles//256+1,1))
        self.energy = 0.5 * mass * (self.particle_velocities.get() ** 2).sum() / num_particles
        print(self.energy)
        MOD_PARA = f"#define test_particle {n_test}\n"
        self.mod = SourceModule(MOD_PARA + MOD_MC)
        self.update_positions = self.mod.get_function("update_positions")
        self.handle_collisions = self.mod.get_function("handle_collisions")
        self.cell_states = self.mod.get_function("cell_states")
        self.num_grids = box_size ** 3
        self.max_grid_size = (num_particles // self.num_grids + 1) * 2
        print("max_grid_size", self.max_grid_size)
        self.pressures = []
        self.pressures_from_momentum = []
        self.Eks = []
        self.collision_mean_rates_o0 = []
        self.collision_mean_rates_o1 = []
        self.collision_mean_rates_o2 = []
        self.processes = {'update_positions':[],'handle_collisions':[],'cell_states':[]}
        self.processes_correct = {'update_positions':num_particles,'handle_collisions':self.num_grids,'cell_states':self.num_grids}
        self.collision_rands = curand((30, self.num_grids*2, self.max_grid_size**2))
        print("pressure_thero", num_particles * T / (box_size ** 3 - num_particles * 2/3*np.pi*d**3))
        print("pressure_thero(r0=d/2)", num_particles * T / (box_size ** 3 - num_particles * 2/3*np.pi*d**3/8))
        for _ in tqdm(range(int(stop/dt))):
            pressure = gpuarray.zeros(1, np.float32)
            process_counts = gpuarray.zeros(3, np.int32)
            grid = gpuarray.zeros((self.num_grids, self.max_grid_size), np.int32)
            grid_sizes = gpuarray.zeros(self.num_grids, np.int32)
            collision_rates_o0 = gpuarray.zeros(self.num_grids, np.float32)
            collision_rates_o1 = gpuarray.zeros(self.num_grids, np.float32)
            collision_rates_o2 = gpuarray.zeros(self.num_grids, np.float32)
            self.update_positions(process_counts, grid, grid_sizes, self.particle_positions, self.particle_velocities, pressure, self.offsets, np.float32(box_size), np.int32(self.max_grid_size), np.float32(mass), np.float32(dt), np.int32(num_particles), np.int32(self.bounded), np.int32(offset_pos), block=(512,1,1), grid=(num_particles//512+1,1))
            pressure *= 2 * mass / 6 / box_size / box_size / dt
            self.pressures.append(pressure.get()[0])
            collision_rand = self.collision_rands[np.random.randint(20)]
            self.handle_collisions(process_counts, grid, grid_sizes, self.particle_positions, self.particle_velocities, collision_rates_o0, collision_rates_o1, collision_rates_o2, np.float32(d), np.int32(box_size), np.int32(self.max_grid_size), np.float32(mass), np.float32(dt), np.int32(num_particles), collision_rand, np.int32(order), np.int32(self.bounded), block=(512,1,1), grid=(self.num_grids//512+1,1))
            self.collision_mean_rates_o0.append(collision_rates_o0.get().mean())
            self.collision_mean_rates_o1.append(collision_rates_o1.get().mean())
            self.collision_mean_rates_o2.append(collision_rates_o2.get().mean())
            Ek = 0.5 * mass * (self.particle_velocities.get() ** 2).sum() / num_particles
            self.Eks.append(Ek)
            self.processes['update_positions'].append(process_counts.get()[0])
            self.processes['handle_collisions'].append(process_counts.get()[1])
            if _ % dt_get_states == 0:
                us = gpuarray.zeros((self.num_grids, 4), np.float32)
                p_tensors = gpuarray.zeros((self.num_grids, 6), np.float32)
                self.cell_states(process_counts, grid, grid_sizes, self.particle_positions, self.particle_velocities, us, p_tensors, np.int32(box_size), np.int32(self.max_grid_size), block=(512,1,1), grid=(self.num_grids//512+1,1))
                self.pressures_from_momentum.append((p_tensors.get()[:,0].sum() + p_tensors.get()[:,1].sum() + p_tensors.get()[:,2].sum())*self.mass/3/box_size**3)
                self.processes['cell_states'].append(process_counts.get()[2])
        self.energy = 0.5 * mass * (self.particle_velocities.get() ** 2).sum() / num_particles
        print(self.energy)
        for k,v in self.processes.items():
            print(k, sum(v)/len(v), 'times per step, should be', self.processes_correct[k])
        
    def out_plot(self, suffix = "", save_pdf=False):
        fig, ax = plt.subplots(3, 3, figsize=(15, 15))
        ax[0,0].scatter(np.linspace(0, self.stop, len(self.collision_mean_rates_o0)), self.collision_mean_rates_o0, s=1)
        ax[0,0].set_title("Collision mean rates order 0")
        ax[0,0].set_xlabel("t")
        ax[0,1].scatter(np.linspace(0, self.stop, len(self.collision_mean_rates_o1)), self.collision_mean_rates_o1, s=1)
        ax[0,1].set_title("Collision mean rates order 1")
        ax[0,1].set_xlabel("t")
        ax[0,2].scatter(np.linspace(0, self.stop, len(self.collision_mean_rates_o2)), self.collision_mean_rates_o2, s=1)
        ax[0,2].set_title("Collision mean rates order 2")
        ax[0,2].set_xlabel("t")
        ax[1,0].hist(0.5 * self.mass * (self.particle_velocities.get() ** 2).sum(-1), bins=200)
        ax[1,0].set_title("Kinetic energy distribution")
        ax[1,0].set_xlabel("E")
        particle_energies = 0.5 * self.mass * (self.particle_velocities.get() ** 2).sum(-1)
        def exp_func(x, a, temp):
            return a * np.exp(-x/temp)
        hist, bins = np.histogram(particle_energies, bins=200)
        bin_widths = np.diff(bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        norm_factor = bin_widths * np.sqrt(bin_centers)
        normalized_hist = hist / norm_factor
        popt, pcov = curve_fit(exp_func, bin_centers, normalized_hist, bounds=([0, 0], [1e8, 2*self.T]))
        fit_values = exp_func(bin_centers, *popt)
        ax[1,1].bar(bin_centers, normalized_hist, width=bin_widths, align='edge', label='Data')
        ax[1,1].plot(bin_centers, fit_values, 'r-', label='Fit: a=%.5f, T=%.5f' % tuple(popt))
        ax[1,1].set_yscale('log')
        ax[1,1].set_title(R"$\mathrm{d}N/\sqrt{E}\mathrm{d}E$")
        ax[1,1].set_xlabel("E")
        ax[1,1].legend()
        ax[1,2].scatter(np.linspace(0, self.stop, len(self.Eks)), self.Eks, s=1)
        ax[1,2].set_title("Kinetic energy")
        ax[1,2].set_xlabel("t")
        ax[2,0].set_title("Pressure")
        # ax[2,0].scatter(np.linspace(0, self.stop, len(self.pressures)), self.pressures, s=1, label='from collision to wall')
        ax[2,0].scatter(np.linspace(0, self.stop, len(self.pressures_from_momentum)), self.pressures_from_momentum, s=1, label='from momentum')
        Ps = []
        meaning = 100
        for i in range(0, len(self.pressures), meaning):
            Ps.append(sum(self.pressures[i:i+meaning]) / meaning)
        ax[2,0].plot(np.linspace(0, self.stop, len(Ps)), Ps, label='pooled Pressure from collision to wall')
        ax[2,0].legend()
        ax[2,1].hist([self.particle_velocities.get()[:,0], self.particle_velocities.get()[:,1], self.particle_velocities.get()[:,2]], bins=200, histtype='step', label=['v_x', 'v_y', 'v_z'])
        ax[2,1].set_title("Velocity distribution")
        ax[2,1].legend()
        ax[2,2].hist([self.particle_positions.get()[:,0], self.particle_positions.get()[:,1], self.particle_positions.get()[:,2]], bins=200, histtype='step', label=['x', 'y', 'z'])
        ax[2,2].set_title("Position distribution")
        ax[2,2].legend()        
        fig.tight_layout()
        plt.show()
        if save_pdf:
            fig.savefig(f"box_{self.box_size}_particles_{self.num_particles}_stop_{self.stop}_dt_{self.dt:.2f}_T_{self.T}_mass_{self.mass}_d_{self.d:.2f}_{'bounded' if self.bounded else ''}_{suffix}.pdf")


if __name__ == "__main__":
    box = Box(box_size=32, stop=1000, dt=0.1, T=1, mass=200, d=0.4, bounded=True, n_test=10)
    box.out_plot()