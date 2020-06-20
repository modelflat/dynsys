from typing import Tuple, Optional, Union, Iterable

# variable is either strictly defined (float) or varied (from, to, number of splits)
# all variables are described as a Grid instance
# no more than 3 variables may vary at once

# x, y, z
# t = (1.0, (-1, 1, 100), (-1, 1, 100))
# _ = Grid(t)
import numpy


class Grid:

    def __init__(self, variables: Iterable):
        self._all = None
        self._all = variables
        self._var = tuple(
            (i, v) if len(v) == 3 else (v[0], v[1:])
            for i, v in enumerate(variables)
            if isinstance(v, Iterable)
        )
        for _, v in self._var:
            _, _, n = v
            if not isinstance(n, int):
                # todo more helpful error message
                raise ValueError("Variables should be defined as (from, to, n)")
        if len(self._var) > 3:
            raise ValueError("Can't vary more than 3 variables at once")

    def init(self, real_type):
        return numpy.array([
            numpy.nan if isinstance(v, Iterable) else float(v)
            for v in self._all
        ], dtype=real_type)

    @property
    def bounds(self):
        return tuple(b for _, b in self._var)

    @property
    def varied(self):
        return tuple(i for i, _ in self._var)

    @property
    def shape(self):
        return tuple(n for _, (_, _, n) in self._var)
