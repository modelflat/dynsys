from mako.template import Template

with open("templates/rk4.mako.cl") as f:
    tpl = Template(f.read())

var_type = "real"
vars_t = "real3"
params_t = "real3"
var_names = ["x", "y", "z", "x_", "y_", "z_"]
equations = [
    ["syst1.eq1", "syst1.eq2", "syst1.eq3"],
    ["syst2.eq1", "syst2.eq2", "syst2.eq3"],
    ["syst3.eq1", "syst3.eq2", "syst3.eq3"],
]
n_systems = len(equations)

def assign(dst, sign, src):
    return "{dst} {sign} {src}".format(
        dst=dst, sign=sign, src=src.format(ct="")
    )

def component_mod(i):
    return ""

def call_equations(time_var, left_side_mod, right_side_mod):
    pre = "\n".join("\tconst {var_type} {name} = {rsm}.{name};".format(
        name=var, var_type=var_type, rsm=right_side_mod
    ) for var in var_names)
    body = "\n".join("\t{lsm}[{eqno}]{ct} = {eq};".format(
        lsm=left_side_mod, eqno=i, ct=component_mod(i), eq=eq
    ) for i, syst in enumerate(equations) for eq in syst)
    return "\n".join(("{", pre, body, "}"))

print(tpl.render(
    vars_t=vars_t,
    params_t=params_t,
    equations=equations,
    n_systems=n_systems,
    assign=assign,
    call_equations=call_equations
))