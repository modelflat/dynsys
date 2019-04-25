from dynsys import SimpleApp, createContextAndQueue
from dynsys.LCE import dummyOption
import numpy
import pyopencl as cl
from mako.template import Template


STRUCT_TEMPLATE = Template("""
struct ${struct_name} {
% for var in fields:
    ${var.signature()};
% endfor
};
""")


FUNCTION_TEMPLATE = Template("""
void ${rule.function_name()}(
    struct ${rule.var_struct_name()}* vars,
    const struct ${rule.const_struct_name()}* pars
) {
% for var in rule.vars:
    const ${var.typename()} ${var.name()} = (*vars).${var.name()};
% endfor
% for par in rule.const_vars:
    const ${par.typename()} ${par.name()} = (*pars).${par.name()};
% endfor
    // EQUATIONS PROVIDED BY USER:
% for var, eq in zip(rule.vars, rule.equations):
    (*vars).${var.name()} = (${var.typename()})(
        ${eq}
    );
% endfor
    // END
}
""")


INIT_TEMPLATE = Template("""
struct ${rule.var_struct_name()} _vars = {
% for i, var in enumerate(rule.vars):
    (${var.typename()})(
        ${rule.initialize(var)}
    ) ${"" if i == len(rule.vars) - 1 else ","}
% endfor
};
const struct ${rule.const_struct_name()} _pars = {
% for i, par in enumerate(rule.const_vars):
    (${par.typename()})(${rule.initialize(par)}) ${"" if i == len(rule.const_vars) - 1 else ","}
% endfor
};
""")


class Var:

    def __init__(self, const_mod, cl_type_name_or_name, name, free):
        if name is None:
            self._name = cl_type_name_or_name
            self._cl_type_name = "real"
        else:
            self._name = name
            self._cl_type_name = cl_type_name_or_name
        self._const_mod = const_mod if const_mod else None
        self._free = free

    def free(self):
        return self._free

    def name(self):
        return self._name

    def typename(self):
        return self._cl_type_name

    def signature(self):
        return "{}{}{} {}".format(self._const_mod if self._const_mod is not None else "",
                                  " " if self._const_mod is not None else "",
                                  self._cl_type_name,
                                  self._name)

    def is_const(self):
        return self._const_mod == "const"

    @staticmethod
    def const(cl_type_name_or_name, name=None, free=False):
        return Var("const", cl_type_name_or_name, name, free)

    @staticmethod
    def mut(cl_type_name_or_name, name=None, free=False):
        return Var(None, cl_type_name_or_name, name, free)

    @staticmethod
    def free_const(cl_type_name_or_name, name=None):
        return Var.const(cl_type_name_or_name, name, free=True)


class EvolutionRule:

    def __init__(self, name, variables, equations, init_rules: dict = None):
        self.name = name
        self.vars = tuple(filter(lambda v: not v.is_const() and not v.free(), variables))
        self.const_vars = tuple(filter(lambda v: v.is_const() and not v.free(), variables))
        self.free_vars = tuple(filter(lambda v: v.free(), variables))
        self.equations = equations
        self.init_rules = init_rules

    def var_struct_name(self):
        return "er_{}_vars_{}".format(self.name,
                                      "".join(map(lambda v: v.name(), self.vars)) or "empty")

    def var_struct_def(self):
        return STRUCT_TEMPLATE.render(
            struct_name=self.var_struct_name(),
            fields=self.vars
        )

    def const_struct_name(self):
        return "er_{}_const_vars_{}".format(self.name,
                                            "".join(map(lambda v: v.name(), self.const_vars)) or "empty")

    def const_struct_def(self):
        return STRUCT_TEMPLATE.render(
            struct_name=self.const_struct_name(),
            fields=self.const_vars
        )

    def struct_init(self):
        if self.init_rules is None:
            return ""
        return INIT_TEMPLATE.render(rule=self)

    def function_name(self):
        return "fn_" + self.name

    def function_def(self):
        return FUNCTION_TEMPLATE.render(rule=self)

    def initialize(self, ent):
        rule = self.init_rules.get(ent, None)
        if rule is not None:
            return rule.render(var=ent)
        return 0.0

    def kernel_iterate(self):
        raise NotImplementedError()

    @staticmethod
    def anon(variables, equations, init_rules):
        name = "anon" + str(numpy.random.randint(0, 1 << 20, dtype=numpy.int32))
        return EvolutionRule(name, variables, equations, init_rules)


class InitializationRule:

    def __init__(self, init_code, **tpl_args):
        self.init_code = Template(init_code)
        self.tpl_args = tpl_args

    def render(self, var):
        return self.init_code.render(var=var, **self.tpl_args)

    @staticmethod
    def const(value):
        return InitializationRule(str(value))

    @staticmethod
    def lerp_worker_id(min_val, max_val, dim):
        return InitializationRule(
            "${min_val} + ((${max_val}) - (${min_val})) * "
            "(${var.typename()})(get_global_id(${dim})) / (${var.typename()})(get_global_size(${dim}))",
            min_val=min_val,
            max_val=max_val,
            dim=dim
        )

    @staticmethod
    def kernel_input(at=None, from_var=None):
        return InitializationRule("in_${(var_override if var_override is not None else var).name()}${index}",
                                  var_override=from_var,
                                  index="" if at is None else "[{}]".format(at))

    @staticmethod
    def kernel_input_at_worker_id(dim):
        return InitializationRule.kernel_input("get_global_id({})".format(dim))


x = Var.mut("x")
y = Var.mut("y")
z = Var.mut("z")
x_ = Var.mut("x_")
y_ = Var.mut("y_")
z_ = Var.mut("z_")
a = Var.const("a")


r = EvolutionRule.anon(
    (x, y, z, x_, y_, z_, a),
    (
        "z_ + x + y",
        "y_ + z - a",
        "x_ + y - z"
    ),
    {
        x:  InitializationRule.const(3.14),
        y:  InitializationRule.lerp_worker_id(-1.0, 1.0, 1),
        z:  InitializationRule.kernel_input(),
        x_: InitializationRule.kernel_input_at_worker_id(0),
        y_: InitializationRule.const(2),
        # z_: InitializationRule.lerp_worker_id("in_z", "in_z", 0)
    }
)

# print(r.const_struct_def())
# print(r.var_struct_def())
# print(r.function_def())

ctx, queue = createContextAndQueue()
src = """
           #define real float
           """ +\
r.const_struct_def() + r.var_struct_def() + r.function_def() +\
"""

kernel void test(int a, const global real* in_x_, real in_y, real in_z) {{
     {} 
}}
""".format(r.struct_init())
print(src)

p = cl.Program(ctx, src).build(options=[dummyOption()])


print(p.get_info(cl.program_info.SOURCE))
