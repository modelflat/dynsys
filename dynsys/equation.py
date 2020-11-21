from typing import Dict, Tuple
from collections import OrderedDict

import numpy
import pyopencl.cltypes as cltypes


_VALID_SYSTEM_TYPES = {"continuous", "discrete"}


class Equation:

    def __init__(self, var, expr):
        self.var = var.strip()
        self.expr = expr.strip()

    @staticmethod
    def from_str(s):
        var, expr = s.split("=", 1)
        return Equation(var, expr)


class EquationSystem:

    def __init__(
            self,
            *equations,
            parameters=tuple(),
            extra_parameters=tuple(),
            kind="discrete",
            before_start=tuple(),
            before_step=tuple(),
            after_step=tuple(),
            real_type=None,
            real_type_name=None,
    ):
        if len(equations) < 1:
            raise ValueError("'equations' cannot be empty")

        if len(equations) > 16:
            raise ValueError("Systems of more than 16 equations are not currently supported")

        if kind not in _VALID_SYSTEM_TYPES:
            raise ValueError("Valid values of 'kind' are {}, got {}", _VALID_SYSTEM_TYPES, kind)

        equations = [
            Equation.from_str(eq) if isinstance(eq, str) else eq for eq in equations
        ]

        self.equations = equations
        self._kind = kind
        self._parameter_names = parameters
        self._variable_names = [eq.var for eq in equations]

        # TODO these should not be hardcoded
        self._real_type = real_type or numpy.float64
        self._real_type_name = real_type_name or "double"

        self.extra_parameters = OrderedDict((name, tp) for name, tp in extra_parameters)
        self.before_step = (before_step,) if isinstance(before_step, str) else before_step
        self.before_start = (before_start,) if isinstance(before_start, str) else before_start
        self.after_step = (after_step,) if isinstance(after_step, str) else after_step

        self.parameter_t = self.parameters_dtype()
        self.variable_t = self.variable_dtype()

    @property
    def dimensions(self):
        return len(self.equations)

    @property
    def variables(self):
        return self._variable_names

    @property
    def parameters(self):
        return self._parameter_names

    @property
    def is_discrete(self):
        return self._kind == "discrete"

    @property
    def is_continuous(self):
        return self._kind == "continuous"

    @property
    def is_represented_by_cl_type(self):
        return self.dimensions in {1, 2, 3, 4, 8, 16}

    @property
    def is_represented_by_geometric_cl_type(self):
        return self.dimensions in {1, 2, 3, 4}

    @property
    def real_type(self):
        return self._real_type

    def _parameters_with_types(self):
        parameters = OrderedDict()
        for name in self.parameters:
            parameters[name] = self._real_type_name
        for name, tp in self.extra_parameters.items():
            parameters[name] = tp
        return parameters

    def parameters_dtype(self):
        return numpy.dtype([(name, cltypes.get_or_register_dtype(tp))
                            for name, tp in self._parameters_with_types().items()], align=False)

    def parameters_value(self, values):
        parameters_with_types = self._parameters_with_types()
        for name in values.keys():
            if name not in parameters_with_types:
                raise ValueError(f"Unknown parameter: '{name}'!")

        # TODO this might be slow
        value = []
        for name, tp in parameters_with_types.items():
            if name not in values:
                raise ValueError(f"Parameter '{name}' is not set!")
            value.append(values[name])

        return numpy.array([tuple(value)], dtype=self.parameter_t)

    def variable_dtype(self):
        return numpy.dtype([(name, self.real_type) for name in self.variables])


class Parameters:

    def __init__(self, **values):
        self._values = values

    def to_cl_object(self, system: EquationSystem):
        return system.parameters_value(self._values)

