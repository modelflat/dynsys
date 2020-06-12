from typing import Tuple


class Grid:

    def __init__(self, x, y=None, z=None):
        self.x_bounds, self.x_units = x
        self.shape = (self.x_units,)
        self.dimensions = 1
        if y is not None:
            self.y_bounds, self.y_units = y
            self.shape = (self.x_units, self.y_units)
            self.dimensions += 1

        if z is not None:
            self.z_bounds, self.z_units = z
            self.shape = (self.x_units, self.y_units, self.z_units)
            self.dimensions += 1
