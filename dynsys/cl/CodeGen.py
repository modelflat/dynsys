VARIABLE_TEMPLATE = r"""
#define VARIABLE _DS_var
#define VARIABLE_TYPE {}
#define VARIABLE_SIGNATURE VARIABLE_TYPE VARIABLE 
"""

PARAMETER_NAME = "_DS_par_%d"
PARAMETER_TEMPLATE = r"""
#define SET_PARAMETER(idx, value) {}
#define PARAMETER_TYPE {}
#define PARAMETERS {}
#define PARAMETERS_SIGNATURE {}
"""

BOUNDS_TEMPLATE = r"""
#define BOUNDS_VAR _DS_bs 
#define BOUNDS_1D {}2 
#define BOUNDS_2D {}4
#define BOUNDS_3D {}8

#if   (DIM == 1)
#define BOUNDS BOUNDS_1D
#elif (DIM == 2)
#define BOUNDS BOUNDS_2D
#elif (DIM == 3)
#define BOUNDS BOUNDS_3D
#endif
"""

IMAGE_BOUNDS_TEMPLATE = r"""
#define IMAGE_BOUNDS_VAR _DS_ibs
#define IMAGE_BOUNDS_2D int2
#define IMAGE_BOUNDS_3D int4

#if   (DIM == 2)
#define IMAGE_BOUNDS IMAGE_BOUNDS_2D
#elif (DIM == 3)
#define IMAGE_BOUNDS IMAGE_BOUNDS_3D
#endif
"""

COMMONS_1D = r"""

#define ID_1D get_global_id(0)

#define SIZE_1D get_global_size(0)

#define TRANSLATE_1D(id, size, bs) \
    ((bs).s0 + (id)*((bs).s0 - (bs).s1)/(size))

#define TRANSLATE_BACK_1D(v, bs, size) \
    (((v) - (bs).s0) / ((bs).s1 - (bs).s0) * (size))

#define TRANSLATE_BACK_INV_1D(v, bs, size) \
    ((size) - TRANSLATE_BACK_1D((v), (bs), (size)))
    
#define IN_RANGE_EX(val, range) ((val) > (range).s0 && (val) < (range).s2)

#define NEAR_1D(a, b, precision) (fabs((a) - (b)) < (precision))

"""

COMMONS_2D = r"""

#define ID_2D (int2)(get_global_id(0), get_global_id(1))

#define ID_Y_INV_2D (int2)(get_global_id(0), get_global_size(1) - get_global_id(1))

#define SIZE_2D (int2)(get_global_size(0), get_global_size(1))

#define TRANSLATE_2D(T, id, size, bs) \
    (T)((bs).s0 + (id).x*((bs).s1 - (bs).s0)/(size).x, (bs).s2 + (id).y*((bs).s3 - (bs).s2)/(size).y)

#define TRANSLATE_INV_Y_2D(T, id, size, bs) \
    (T)((bs).s0 + (id).x*((bs).s1 - (bs).s0)/(size).x, (bs).s2 + ((size).y - (id).y)*((bs).s3 - (bs).s2)/((size).y))

#define TRANSLATE_BACK_2D(T, v, bs, size) \
    (T)(((v).x - (bs).s0)/((bs).s1 - (bs).s0)*(size).x, \
        ((v).y - (bs).s2)/((bs).s3 - (bs).s2)*(size).y )

#define TRANSLATE_BACK_INV_Y_2D(T, v, bs, size) \
    (T)(((v).x - (bs).s0)/((bs).s1 - (bs).s0)*(size).x, \
        (size).y - ((v).y - (bs).s2)/((bs).s3 - (bs).s2)*(size).y )
        
#define CONVERT_SPACE_TO_COORD_2D(val) convert_int2_rtz(val)

#define VALID_POINT_2D(area, point) \
    (point.x >= 0 && point.y >= 0 && point.x < area.x && point.y < area.y)


"""

COMMONS_3D = r"""
#define ID_3D (int3)(get_global_id(0), get_global_id(1), get_global_id(2))

#define ID_Y_INV_3D (int3)(get_global_id(0), get_global_size(1) - get_global_id(1), get_global_id(2))

#define SIZE_3D (int3)(get_global_size(0), get_global_size(1), get_global_size(2))

#define TRANSLATE_3D(T, id, size, bs) \
    (T)((bs).s0 + (id).x*((bs).s1 - (bs).s0)/(size).x, \
        (bs).s2 + (id).y*((bs).s3 - (bs).s2)/(size).y, \
        (bs).s4 + (id).z*((bs).s5 - (bs).s4)/(size).z )

#define TRANSLATE_INV_Y_3D(T, id, size, bs) \
    (T)((bs).s0 + (id).x*((bs).s1 - (bs).s0)/(size).x,\
        (bs).s2 + ((size).y - (id).y)*((bs).s3 - (bs).s2)/((size).y), \
        (bs).s4 + (id).z*((bs).s5 - (bs).s4)/(size).z )

#define TRANSLATE_BACK_3D(T, v, bs, size) \
    (T)(((v).x - (bs).s0)/((bs).s1 - (bs).s0)*(size).x, \
        ((v).y - (bs).s2)/((bs).s3 - (bs).s2)*(size).y, \
        ((v).z - (bs).s4)/((bs).s5 - (bs).s4)*(size).z)

#define TRANSLATE_BACK_INV_Y_3D(T, v, bs, size) \
    (T)(((v).x - (bs).s0)/((bs).s1 - (bs).s0)*(size).x, \
        (size).y - ((v).y - (bs).s2)/((bs).s3 - (bs).s2)*(size).y, \
        ((v).z - (bs).s4)/((bs).s5 - (bs).s4)*(size).z)

#define CONVERT_SPACE_TO_COORD_3D(val) convert_int4_rtz((real4)(val, 0.0))

#define VALID_POINT_3D(area, point) \
    (point.x >= 0 && point.y >= 0 && point.z >= 0 && point.x < area.x && point.y < area.y && point.z < area.z)


"""

COMMON_SOURCE = COMMONS_1D + COMMONS_2D + COMMONS_3D + r"""

#if (!defined(DIM) || DIM < 1 || DIM > 3)
#error 'DIM' should be defined to be either 1, 2 or 3
#endif

#if   (DIM == 1) 

#define ID ID_1D
#define SIZE SIZE_1D

#elif (DIM == 2)

#define COORD_TYPE int2
#define COORD_TYPE_EXPORT COORD_TYPE
#define IMAGE_TYPE image2d_t

#define ID ID_2D
#define ID_Y_INV ID_Y_INV_2D
#define SIZE SIZE_2D

#elif (DIM == 3)

#define COORD_TYPE int3
#define COORD_TYPE_EXPORT int4
#define IMAGE_TYPE image3d_t

#define ID ID_3D
#define ID_Y_INV ID_Y_INV_3D
#define SIZE SIZE_3D

#endif 

#if   (DIM == 2)
#define TRANSLATE TRANSLATE_2D
#elif (DIM == 3)
#define TRANSLATE TRANSLATE_3D
#endif

#if   (DIM == 2)
#define TRANSLATE_INV_Y TRANSLATE_INV_Y_2D
#elif (DIM == 3)
#define TRANSLATE_INV_Y TRANSLATE_INV_Y_3D
#endif

#if   (DIM == 2)
#define TRANSLATE_BACK TRANSLATE_BACK_2D
#elif (DIM == 3)
#define TRANSLATE_BACK TRANSLATE_BACK_3D
#endif

#if   (DIM == 2)
#define TRANSLATE_BACK_INV_Y TRANSLATE_BACK_INV_Y_2D
#elif (DIM == 3)
#define TRANSLATE_BACK_INV_Y TRANSLATE_BACK_INV_Y_3D
#endif

#if   (DIM == 2)
#define VALID_POINT VALID_POINT_2D
#elif (DIM == 3)
#define VALID_POINT VALID_POINT_3D
#endif

#if   (DIM == 2)
#define CONVERT_SPACE_TO_COORD CONVERT_SPACE_TO_COORD_2D
#elif (DIM == 3)
#define CONVERT_SPACE_TO_COORD CONVERT_SPACE_TO_COORD_3D
#endif



#define DEFAULT_ENTITY_COLOR (float4)(0.0, 0.0, 0.0, 1.0) // black

float3 hsv2rgb(float3);

float3 hsv2rgb(float3 hsv) {
    const float c = hsv.y * hsv.z;
    const float x = c * (1 - fabs(fmod( hsv.x / 60, 2 ) - 1));
    float3 rgb;
    if      (0 <= hsv.x && hsv.x < 60) {
        rgb = (float3)(c, x, 0);
    } else if (60 <= hsv.x && hsv.x < 120) {
        rgb = (float3)(x, c, 0);
    } else if (120 <= hsv.x && hsv.x < 180) {
        rgb = (float3)(0, c, x);
    } else if (180 <= hsv.x && hsv.x < 240) {
        rgb = (float3)(0, x, c);
    } else if (240 <= hsv.x && hsv.x < 300) {
        rgb = (float3)(x, 0, c);
    } else {
        rgb = (float3)(c, 0, x);
    }
    return (rgb + (hsv.z - c));
}


"""


def generateParameterCode(typeConfig, paramCount: int) -> str:
    if paramCount > 8:
        # sanity check
        raise ValueError("Supported dimensions are 1-8 (%d requested)" % (paramCount,))
    names = [PARAMETER_NAME % (i,) for i in range(paramCount)]
    paramType = typeConfig.paramTypeName
    values = ", ".join(names)
    signatures = ", ".join([paramType + " " + name for name in names])
    setter = "{\\\n\t" + "; \\\n\t".join([
        "if ((idx) == %d) %s = (value)" % (i, name) for i, name in enumerate(names)
    ]) + ";}"
    return PARAMETER_TEMPLATE.format(setter, paramType, values, signatures)


def generateVariableCode(typeConfig, varCount: int) -> str:
    if varCount > 3:
        # sanity check
        raise ValueError("Supported dimensions are 1-3 (%d requested)" % (varCount,))
    return VARIABLE_TEMPLATE.format(typeConfig.varTypeName + "{}".format(varCount if varCount > 1 else ""))


def generateBoundsCode(typeConfig, dims: int) -> str:
    if dims > 3 or dims < 1:
        # sanity check
        raise ValueError("Supported dimensions for bounds are 1, 2 and 3 (%d requested)" % (dims,))
    return BOUNDS_TEMPLATE.format(*[typeConfig.boundsTypeName, ]*3)


def generateImageBoundsCode(dims: int) -> str:
    if dims > 3 or dims < 2:
        # sanity check
        raise ValueError("Supported dimensions for image bounds are 2 and 3 (%d requested)" % (dims,))
    return IMAGE_BOUNDS_TEMPLATE


def makeSource(*args, typeConfig):
    return typeConfig.cl() + "\n" + COMMON_SOURCE + "\n" + "\n".join(args)
