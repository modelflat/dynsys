from mako.template import Template

BOUNDS = r"""
<%
    define_prefix = "_DYNSYS__"
    var_prefix = "_dynsys__"
    
    def bound_name(i):
        return "b{}".format(i)
        
    def image_bound_name(i):
        return "ib{}".format(i)
    
    def const(cond):
        if cond: return "const "
        return ""
%>

<%def name="gen_params()">
// system parameters
    % for i, val in enumerate(param_names):
    ${const(True)}
// end system parameters
</%def>

<%def name="gen_bounds()">
// plane bounds
    % for i, val in enumerate(bounds):
    ${const(True)}${bounds_type} ${var_prefix}${bound_name(i)},
    % endfor
// end plane bounds
</%def>

<%def name="gen_image_bounds()">
// image bounds
???
// end image bounds
</%def>

kernel void phase(
    ${gen_params()}
    ${gen_bounds()}
    const int skip,
    const int iterations,
    write_only IMAGE_TYPE result
) {
    const COORD_TYPE id = ID;
    VARIABLE_TYPE point = TRANSLATE_INV_Y(VARIABLE_TYPE, id, SIZE, bounds);
    
    for (int i = 0; i < skip; ++i) {
        point = userFn(point, PARAMETERS);
    }
    
    for (int i = skip; i < iterations; ++i) {
        point = userFn(point, PARAMETERS);
        const COORD_TYPE_EXPORT coord = CONVERT_SPACE_TO_COORD(TRANSLATE_BACK_INV_Y(VARIABLE_TYPE, point, bounds, image_bounds));
        
        if (VALID_POINT(image_bounds, coord)) {
#ifdef DYNAMIC_COLOR
            const float ratio = (float)(i - skip) / (float)(iterations - skip);
            write_imagef(result, coord, (float4)(hsv2rgb((float3)( 240.0 * (1.0 - ratio), 1.0, 1.0)), 1.0));
#else
            write_imagef(result, coord, DEFAULT_ENTITY_COLOR);
#endif
        }
    }
}
"""


print(
    Template(text=BOUNDS).render(
        compile_time_bounds=True,
        bounds_const_in_runtime=True,
        bounds_type="double",
        bounds=(0.0, 1.0),
        compile_time_image_bounds=False,
        infer_image_bounds=True,
        image_bounds_type="size_t",
        image_bounds=(512, 512)
    ).strip()
)
