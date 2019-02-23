from mako.template import Template

src = r"""
<%
    def comma(cond):
        if cond: return ","
        return ""
        
    def const(cond):
        if cond: return "const "
        return ""
%>

<%def name="gen_par_signatures(end_comma=False)"> 
% for i, par in enumerate(parameters):
    const ${parameter_type} ${par}${comma(end_comma or i != len(parameters) - 1)}
% endfor
</%def>

<%def name="define_user_system(user_system_name)">
// definition of ${user_system_name}
inline void ${user_system_name}(
% for i, var in enumerate(variables):
    __private ${variable_type}* _${var},
% endfor
    ${gen_par_signatures(end_comma=False)}
) {
% for var in variables:
    const ${variable_type} ${var} = *(_${var});
% endfor
    // user equations:
% for var, eq in zip(variables, equations):
    *(_${var}) = (${variable_type})( ${eq} ); 
% endfor
}
// end definition of ${user_system_name}
</%def>

<%def name="call_user_system(user_system_name)">
    ${user_system_name}(${", ".join(map(lambda var: "&{}".format(var), variables))}, ${", ".join(parameters)});
</%def>

<%def name="gen_kernel_args()">
<% knl_args = list(filter(lambda d: d.source == "kernel_arg", var_name_map)) %>
% for i, name_def in enumerate(knl_args):
    ${name_def.signature()}${comma(i != len(knl_args) - 1)}
% endfor
</%def>

<%def name="gen_initialization()">
<% init_required = list(filter(lambda d: d.source == "worker_id", var_name_map)) %>
% for name_def in init_required:
    ${name_def.initialization()};
% endfor
</%def>


${define_user_system(user_system_name)}

kernel void phase (
${gen_kernel_args()}
) {
    ${gen_initialization()}
    
    for (int i = 0; i < skip; ++i) {
        ${call_user_system(user_system_name)}
    }
    
    for (int i = skip; i < iter; ++i) {
        ${call_user_system(user_system_name)}
    }
}
"""


def assemble_system_function(equations, variables, parameters, var_name_map):
    return Template(text=src).render(
        variable_type="float",
        variables=variables,
        parameter_type="float",
        parameters=parameters,
        equations=equations,
        user_system_name="user_system",
        var_name_map=var_name_map,
    ).strip()


def _interpolate(type, from_val, from_size, to_min, to_max, invert):
    return "({type})({to_min} + ({to_max} - {to_min}) * ({invert} ({type})({from_val}) / ({type})({from_size})))"\
        .format(
        type=type,
        from_val=from_val,
        from_size=from_size,
        to_min=to_min,
        to_max=to_max,
        invert="1 - " if invert else ""
    )


def bound_names(name):
    return "{}_min".format(name), "{}_max".format(name)


def interpolate_global_dim(type, dim, name, invert):
    to_min, to_max = bound_names(name)
    from_val, from_size = "get_global_id({})".format(dim), "get_global_size({})".format(dim)
    return _interpolate(type, from_val, from_size, to_min, to_max, invert)


INIT_STRATEGIES = {
    "uniform_grid_global":
        lambda type, name, dim, invert: interpolate_global_dim(type, dim, name, invert),
}


class NameDef:

    def __init__(self, name, type, source, is_const, init_strategy=None, init_strategy_args=None):
        self.name = name
        self.type = type
        self.source = source
        self.is_const = is_const
        self.init_strategy = init_strategy
        self.init_strategy_args = () if init_strategy_args is None else tuple(init_strategy_args)

    def _const(self):
        if self.is_const: return "const "
        return ""

    def _init_value(self):
        if self.init_strategy is None:
            raise RuntimeError("No init strategy set for value of source {}".format(self.source))
        return INIT_STRATEGIES[self.init_strategy](self.type, self.name, *self.init_strategy_args)

    def signature(self):
        if self.source == "worker_id":
            raise RuntimeError("Source type '{}' does not need signature".format(self.source))
        return "{}{} {}".format(self._const(), self.type, self.name)

    def initialization(self):
        if self.source == "kernel_arg":
            raise RuntimeError("Source type '{}' does not support initialization".format(self.source))
        return "{}{} {} = {}".format(self._const(), self.type, self.name, self._init_value())


var_type = "float"
bounds_type = "float"
par_type = "float"


print(assemble_system_function(
    [ # equations
        "-z - y",
        "b + (x - r)*y",
        "x + a * z"
    ],
    [ # variable names
        "x", "y", "z"
    ],
    [ # parameter names
        "a", "b", "r"
    ],
    ( # where names are coming from TODO generate this
        NameDef("x", var_type, "worker_id", is_const=False,
                init_strategy="uniform_grid_global", init_strategy_args=(0, False)),
        NameDef("y", var_type, "worker_id", is_const=False,
                init_strategy="uniform_grid_global", init_strategy_args=(1, True)),
        NameDef("z", var_type, "kernel_arg", is_const=False),
        NameDef("x_min", bounds_type, "kernel_arg", is_const=True),
        NameDef("x_max", bounds_type, "kernel_arg", is_const=True),
        NameDef("y_min", bounds_type, "kernel_arg", is_const=True),
        NameDef("y_max", bounds_type, "kernel_arg", is_const=True),
        NameDef("a", par_type, "kernel_arg", is_const=True),
        NameDef("b", par_type, "kernel_arg", is_const=True),
        NameDef("r", par_type, "kernel_arg", is_const=True),

        NameDef("skip", "int", "kernel_arg", is_const=True),
        NameDef("iter", "int", "kernel_arg", is_const=True)
    )
))

"""
What to support:

raw_value: value known at compile time / variable name.
uniform_grid: value interpolated from execution dimensions.

"""
