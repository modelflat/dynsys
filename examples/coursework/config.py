## general parameters

# range for varying `h`
h_bounds = (-2, 2)

# range for varying `alpha`
alpha_bounds = (0.0, 1.0)

# `c` constant
C = complex(-0.5, 0.5)

## phase plot params

# space bounds
phase_shape = (-2, 2, -2, 2)

# phase plot image shape
phase_image_shape = (512, 512)

# skip iters on phase plot
phase_skip = 128

# iters on phase plot
phase_iter = 128

# grid size for phase plot
phase_grid_size = 2


## parameter map params

# image size
param_map_image_shape = (512, 512)

# skip iters on param map
param_map_skip = 128

# iters on param map
param_map_iter = 128

# same point detection tol
param_map_tolerance = 1e-8

# starting point for param map
param_map_z0 = complex(0.01, 0.01)
