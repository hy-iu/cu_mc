import numpy as np
import warp as wp
 
# 初始化Warp环境
wp.init()
 
# 可视化相关模块
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython import display

@wp.func
def contact_force(
    n: wp.vec3,    # 接触法线
    v: wp.vec3,    # 相对速度
    c: float,      # 穿透深度
    k_n: float,    # 法向刚度
    k_d: float,    # 阻尼系数
    k_f: float,    # 摩擦系数
    k_mu: float,   # 摩擦系数
) -> wp.vec3:
    # 计算法向分量
    vn = wp.dot(n, v)
    jn = c * k_n
    jd = min(vn, 0.0) * k_d
    
    # 接触力
    fn = jn + jd
    
    # 摩擦力计算
    vt = v - n * vn
    vs = wp.length(vt)
    
    if vs > 0.0:
        vt = vt / vs
    
    # 库仑摩擦条件
    ft = wp.min(vs * k_f, k_mu * wp.abs(fn))
    
    return -n * fn - vt * ft

@wp.kernel
def update(
    grid: wp.uint64,          # 哈希网格ID
    particle_x: wp.array(dtype=wp.vec3),  # 粒子位置
    particle_v: wp.array(dtype=wp.vec3),  # 粒子速度
    particle_f: wp.array(dtype=wp.vec3),  # 粒子受力
    radius: float,            # 粒子半径
    k_contact: float,         # 接触刚度
    k_damp: float,            # 阻尼系数
    k_friction: float,        # 摩擦系数
    k_mu: float,              # 摩擦系数
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)  # 获取粒子在网格中的ID
    
    x = particle_x[i]
    v = particle_v[i]
    f = wp.vec3()
    
    # 地面接触处理
    n = wp.vec3(0.0, 1.0, 0.0)  # 地面法线
    c = wp.dot(n, x)             # 与地面距离
    
    if c < 0.02:  # 地面粘附阈值
        f = f + contact_force(n, v, c, k_contact, k_damp, 100.0, 0.5)
    
    # 粒子间接触处理
    neighbors = wp.hash_grid_query(grid, x, radius * 5.0)  # 邻域查询
    
    for index in neighbors:
        if index != i:
            # 计算粒子间距离
            n = x - particle_x[index]
            d = wp.length(n)
            err = d - radius * 2.0
            
            if err <= 0.0075:  # 粒子间粘附阈值
                n = n / d
                vrel = v - particle_v[index]
                f = f + contact_force(n, vrel, err, k_contact, k_damp, k_friction, k_mu)
    
    particle_f[i] = f

@wp.kernel
def integrate(
    x: wp.array(dtype=wp.vec3),  # 位置
    v: wp.array(dtype=wp.vec3),  # 速度
    f: wp.array(dtype=wp.vec3),  # 力
    gravity: wp.vec3,            # 重力
    dt: float,                   # 时间步长
    inv_mass: float,             # 质量倒数
):
    tid = wp.tid()
    v_new = v[tid] + f[tid] * inv_mass * dt + gravity * dt
    x_new = x[tid] + v_new * dt
    v[tid] = v_new
    x[tid] = x_new

# 模拟参数
num_frames = 200          # 总帧数
fps = 60                  # 帧率
sim_substeps = 64         # 每帧子步数
frame_dt = 1.0 / fps      # 帧间隔
sim_dt = frame_dt / sim_substeps  # 模拟步长
 
# 物理参数
point_radius = 0.1        # 粒子半径
inv_mass = 64.0           # 质量倒数
k_contact = 8e3           # 接触刚度
k_damp = 2.0              # 阻尼系数
k_friction = 1.0          # 摩擦系数
k_mu = 1e5                # 摩擦系数
 
# 初始化粒子系统
points = create_particle_grid(8, 32, 8, (0.0, 0.5, 0.0), point_radius, 0.1)
velocities = wp.array(((0.0, 0.0, 15.0),) * len(points), dtype=wp.vec3)
forces = wp.empty_like(points)
 
# 初始化哈希网格
grid = wp.HashGrid(128, 128, 128)
grid_cell_size = point_radius * 5.0

# 初始化渲染器
renderer = wp.Renderer()
image = wp.zeros(shape=(720, 1280, 4), dtype=wp.float32)
renderer.init()

renders = []
for frame in range(num_frames):
    # 重建哈希网格
    grid.build(points, grid_cell_size)
    
    # 子步模拟
    for _ in range(sim_substeps):
        # 计算力
        wp.launch(
            kernel=update,
            dim=points.shape,
            inputs=(
                grid.id,
                points,
                velocities,
                forces,
                point_radius,
                k_contact,
                k_damp,
                k_friction,
                k_mu,
            ),
        )
        
        # 积分
        wp.launch(
            kernel=integrate,
            dim=points.shape,
            inputs=(
                points,
                velocities,
                forces,
                (0.0, -9.8, 0.0),
                sim_dt,
                inv_mass,
            ),
        )
    
    # 渲染当前帧
    renderer.begin_frame(frame / num_frames)
    renderer.render_points(
        points=points.numpy(),
        radius=point_radius,
        name="points",
        colors=(0.8, 0.3, 0.2),
    )
    renderer.end_frame()
    
    # 存储渲染结果
    renderer.get_pixels(image, split_up_tiles=False, mode="rgb")
    renders.append(wp.clone(image, device="cpu", pinned=True))
 
wp.synchronize()

# 设置Matplotlib
fig = plt.figure(figsize=(10, 7.5))
img = plt.imshow(renders[0], animated=True)
plt.axis('off')
 
# 创建动画
anim = matplotlib.animation.FuncAnimation(
    fig,
    lambda frame: img.set_data(renders[frame]),
    frames=num_frames,
    interval=(1.0 / fps) * 1000.0,
)
 
display.display(display.HTML(anim.to_html5_video()))
plt.close()
