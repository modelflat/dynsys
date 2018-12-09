s = r"""

#define MAP_VALUE_TO_SIZE(value, bounds, size) \
    (float3)( \
        ((value).x - (bounds).s0)/((bounds).s1 - (bounds).s0)*(size).x, \
        ((value).y - (bounds).s2)/((bounds).s3 - (bounds).s2)*(size).y, \
        ((value).z - (bounds).s4)/((bounds).s5 - (bounds).s4)*(size).z  \
    )
    
#define MAP_VALUE_TO_IMAGE(value, bounds, image) \
    (float3)( \
        ((value).x - (bounds).s0)/((bounds).s1 - (bounds).s0)*get_image_width (image), \
        ((value).y - (bounds).s2)/((bounds).s3 - (bounds).s2)*get_image_height(image), \
        ((value).z - (bounds).s4)/((bounds).s5 - (bounds).s4)*get_image_depth (image)  \
    )

#define WITHIN(bounds, value) \
    (bounds.s0 <= value.x && bounds.s1 > value.x &&\
     bounds.s2 <= value.y && bounds.s3 > value.y &&\
     bounds.s4 <= value.z && bounds.s5 > value.z)

#define CONVERT_TO_COORD(value) (int4)(convert_int3_rtz(value), 0)

#define FLATTEN_COORD(coord, size) \
    (coord).s2*(size).s1*(size).s0 + (coord).s1*(size).s0 + (coord).s0

"""