Scalar Types
- `warp.int8`, `warp.uint8`, `warp.int16`, `warp.uint16`
- `warp.int32`, `warp.uint32`, `warp.int64`, `warp.uint64`
- `warp.float16`, `warp.float32`, `warp.float64`
- `warp.bool`

Vector Types
- `warp.vec2b`, `warp.vec2ub`, `warp.vec2s`, `warp.vec2us`, `warp.vec2i`, `warp.vec2ui`, `warp.vec2l`, `warp.vec2ul`, `warp.vec2h`, `warp.vec2f`, `warp.vec2d`
- `warp.vec3b`, `warp.vec3ub`, `warp.vec3s`, `warp.vec3us`, `warp.vec3i`, `warp.vec3ui`, `warp.vec3l`, `warp.vec3ul`, `warp.vec3h`, `warp.vec3f`, `warp.vec3d`
- `warp.vec4b`, `warp.vec4ub`, `warp.vec4s`, `warp.vec4us`, `warp.vec4i`, `warp.vec4ui`, `warp.vec4l`, `warp.vec4ul`, `warp.vec4h`, `warp.vec4f`, `warp.vec4d`

Matrix Types
- `warp.mat22h`, `warp.mat22f`, `warp.mat22d`
- `warp.mat33h`, `warp.mat33f`, `warp.mat33d`
- `warp.mat44h`, `warp.mat44f`, `warp.mat44d`

Quaternion & Transform Types
- `warp.quath`, `warp.quatf`, `warp.quatd`
- `warp.transformh`, `warp.transformf`, `warp.transformd`

Spatial Types
- `warp.spatial_vectorh`, `warp.spatial_vectorf`, `warp.spatial_vectord`
- `warp.spatial_matrixh`, `warp.spatial_matrixf`, `warp.spatial_matrixd`

Generic Types
- `warp.Int`, `warp.Float`, `warp.Scalar`, `warp.Vector`, `warp.Matrix`, `warp.Quaternion`, `warp.Transformation`, `warp.Array`

---

Scalar Math
- `min`, `max`, `clamp`, `abs`, `sign`, `step`, `nonzero`
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`
- `sinh`, `cosh`, `tanh`
- `sqrt`, `cbrt`, `log`, `log2`, `log10`, `exp`, `pow`
- `degrees`, `radians`
- `round`, `rint`, `trunc`, `floor`, `ceil`, `frac`
- `isfinite`, `isnan`, `isinf`

Vector & Matrix Math
- `dot`, `ddot`, `argmin`, `argmax`, `outer`, `cross`, `skew`
- `length`, `length_sq`, `normalize`, `transpose`, `inverse`, `determinant`, `trace`
- `diag`, `get_diag`, `cw_mul`, `cw_div`
- `vector`, `matrix`, `matrix_from_cols`, `matrix_from_rows`, `identity`
- `svd2`, `svd3`, `qr3`, `eig3`

Quaternion Math
- `quaternion`, `quat_identity`, `quat_from_axis_angle`, `quat_to_axis_angle`
- `quat_from_matrix`, `quat_rpy`, `quat_inverse`, `quat_rotate`, `quat_rotate_inv`
- `quat_slerp`, `quat_to_matrix`

Transformation Math
- `transformation`, `transform_identity`, `transform_get_translation`, `transform_get_rotation`
- `transform_multiply`, `transform_point`, `transform_vector`, `transform_inverse`

Spatial Math
- `spatial_vector`, `spatial_adjoint`, `spatial_dot`, `spatial_cross`, `spatial_cross_dual`
- `spatial_top`, `spatial_bottom`, `spatial_jacobian`, `spatial_mass`

Tile Primitives
- `tile_zeros`, `tile_ones`, `tile_arange`, `tile_load`, `tile_store`, `tile_atomic_add`
- `tile_view`, `tile_assign`, `tile`, `untile`, `tile_transpose`, `tile_broadcast`
- `tile_sum`, `tile_min`, `tile_max`, `tile_reduce`, `tile_map`, `tile_diag_add`
- `tile_matmul`, `tile_fft`, `tile_ifft`, `tile_cholesky`, `tile_cholesky_solve`

Utility Functions
- `mlp`, `reversed`, `printf`, `print`, `breakpoint`, `tid`, `select`, `where`
- `atomic_add`, `atomic_sub`, `atomic_min`, `atomic_max`
- `lerp`, `smoothstep`, `expect_near`, `len`

Geometry & Volume
- `bvh_query_aabb`, `bvh_query_ray`, `bvh_query_next`
- `mesh_query_point`, `mesh_query_ray`, `mesh_query_aabb`, `mesh_eval_position`, `mesh_eval_velocity`
- `volume_sample`, `volume_lookup`, `volume_store`, `volume_sample_grad`, ...

Random
- `random_init(seed[, offset])`, `randi(state[, low, high])`, `randu(state[, low, high])`, `randf(state[, low, high])`
- `randn(state) → float`, `sample_cdf(state: uint32, cdf: Array[float32]) → int`
- `sample_triangle`, `sample_unit_ring`, `sample_unit_disk`, `sample_unit_sphere`, `sample_unit_sphere_surface`, `sample_unit_hemisphere`, `sample_unit_hemisphere_surface`, `sample_unit_square`, `sample_unit_cube`
- `poisson(state: uint32, lam: float32) → uint32`, `noise(state: uint32, x: float32 / xy: vec2f / xyz: vec3f / xyzt: vec4f) → float`, `pnoise(state: uint32, x: float32, px: int32 / <2d>, ...) → float`, `curlnoise(...)`

Other
- `lower_bound`, `bit_and`, `bit_or`, `bit_xor`, `lshift`, `rshift`, `invert`

Operators
- `add`, `sub`, `mul`, `mod`, `div`, `floordiv`, `pos`, `neg`, `unot`

code-generation
- `static(expr: Any) → Any`
