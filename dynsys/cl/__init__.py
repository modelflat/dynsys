import os
import platform

import numpy
import pyopencl as cl
from mako.lookup import TemplateLookup
from mako.template import Template

_DEBUG_PRINT_CODE = os.environ.get("DYNSYS_DEBUG_PRINT_RENDERED_CODE") == "1"


cl_template_directory = os.path.dirname(os.path.abspath(__file__))


def ocl_compiler_env():
    # compilation cache doesn't work on Mac OS
    if platform.system() == "Darwin":
        os.environ["PYOPENCL_NO_CACHE"] = "1"
    # to provide more info in logs
    os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"


def load_template(name):
    return Template(
        filename=os.path.join(cl_template_directory, name),
        lookup=TemplateLookup([cl_template_directory,]),
        input_encoding="utf-8"
    )


def cleanup_code(source):
    cleaned_up = []
    prev_is_empty = True
    for line in source.splitlines(keepends=True):
        if not line.lstrip():
            if not prev_is_empty:
                prev_is_empty = True
                cleaned_up.append(line)
        else:
            if not (line.startswith("//>") or line.startswith("//<")):
                prev_is_empty = False
                cleaned_up.append(line)
    return "".join(cleaned_up)


def assemble(target, **kwargs):
    already_included = set()

    def mako_include_guard(path):
        nonlocal already_included
        if path not in already_included:
            already_included.add(path)
            return False
        return True

    return cleanup_code(target.render(mako_include_guard=mako_include_guard, **kwargs))


def make_type(ctx, type_name, type_desc, device=None):
    """
    :return: numpy.dtype instance and CL code generated for given type.
    """
    import pyopencl.tools
    dtype, cl_decl = cl.tools.match_dtype_to_c_struct(
        device or ctx.devices[0], type_name, numpy.dtype(type_desc), context=ctx
    )
    type_def = cl.tools.get_or_register_dtype(type_name, dtype)
    return cl_decl, type_def


class Type:

    _SCALAR_TYPES = [
        "char", "uchar", "short", "ushort", "int",
        "uint", "long", "ulong", "float", "double",
    ]

    _KNOWN_TYPES = {
        *_SCALAR_TYPES,
        *[n + str(cnt) for n in _SCALAR_TYPES for cnt in [2, 3, 4, 8, 16]]
    }

    def __init__(self, name: str):
        if not isinstance(name, str) or name not in self._KNOWN_TYPES:
            raise ValueError("name should be one of {}".format(self._KNOWN_TYPES))
        self.name = name


class WithProgram:

    DEFAULT_OPTIONS = (
        "-cl-std=CL1.2",
        "-cl-no-signed-zeros",
        "-cl-single-precision-constant",
    )

    def __init__(self, ctx: cl.Context = None):
        self.ctx = ctx
        self.program = None
        self._options = []
        self._defines = dict()
        self._template_variables = dict()

    def src(self, **kwargs):
        """
        Provides the source code for the underlying OpenCL program. Override this.
        """
        raise NotImplementedError()

    def compile(
            self,
            ctx: cl.Context = None,
            options: tuple = None,
            defines: dict = None,
            template_variables: dict = None
    ):
        """
        Compiles a program composed from ``.src()``. Compilation only happens if one of the following happens:

        * Program has not been compiled yet;

        * OpenCL context has changed since the time it was last compiled;

        * One of the ``options``, ``defines`` or ``template_variables`` has changed since last run.

        :param ctx: OpenCL context to compile this program for
        :param options: options to pass to OpenCL compiler
        :param defines: define options to pass to OpenCL compiler
        :param template_variables: template variables to pass to Mako

        :returns whether the program has been recompiled
        """
        if self.ctx is None:
            self.ctx = ctx

        if ctx is None:
            ctx = self.ctx

        options = options or []
        defines = defines or dict()
        template_variables = template_variables or dict()

        if _DEBUG_PRINT_CODE:
            print("Defines updated: ", defines != self._defines)
            print("Template variables updated: ", template_variables != self._template_variables)
            print("Options updated", options != self._options)

        args_changed = (defines != self._defines
                        or template_variables != self._template_variables
                        or options != self._options)

        if args_changed or self.ctx.int_ptr != ctx.int_ptr or self.program is None:
            ocl_compiler_env()

            self.ctx = ctx
            self._defines = defines
            self._template_variables = template_variables
            self._options = options

            source_code = self.src(**self._template_variables)

            if _DEBUG_PRINT_CODE:
                print(source_code)

            for k, v in self._defines.items():
                options.append(f"-D{k}={v}")

            self.program = cl.Program(self.ctx, source_code) \
                .build(devices=self.ctx.devices,
                       options=[*self.DEFAULT_OPTIONS, *self._options])

            return True

        return False
