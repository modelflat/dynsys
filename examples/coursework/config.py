## general parameters

# range for varying `h`
h_bounds = (1.5, 6)

# range for varying `alpha`
alpha_bounds = (0.0, 1.0)

# `c` constant
C = complex(-0.5, 0.5)


## parameter map params

# image size
param_map_image_shape = (512, 512)

# skip iters on param map
param_map_skip = 64

# iters on param map
param_map_iter = 128

# same point detection tol
param_map_tolerance = 1e-4

# starting point for param map
param_map_z0 = complex(0.01, 0.01)

# enable phase space selection
param_map_select_z0_from_phase = True

# resolution
param_map_resolution = 1


param_map_draw_on_select = True


param_map_lossless = False


## phase plot params

# space bounds
phase_shape = (-1, 1, -1, 1)

# phase plot image shape
phase_image_shape = (512, 512)

# skip iters on phase plot
phase_skip = param_map_skip * 64

# iters on phase plot
phase_iter = 1 << 16

# grid size for phase plot
phase_grid_size = 2

# z0 to use when in single-point mode
phase_z0 = None


## basins params

basins_image_shape = phase_image_shape

basins_resolution = 4

basins_skip = param_map_skip + param_map_iter - 1


## bif tree

bif_tree_skip = 1 << 10
bif_tree_iter = 1 << 10
bif_tree_z0 = complex(-0.1, 0.1)
