COMMON_SOURCE = """

#define ID_2D (int2)(get_global_id(0), get_global_id(1))
#define ID_2D_Y_INV (int2)(get_global_id(0), get_global_size(1) - get_global_id(1))

#define SIZE_2D (int2)(get_global_size(0), get_global_size(1))

#define TRANSLATE(id, size, min_, max_) \
    ((min_) + (id)*((max_) - (min_))/(size))
    
#define TRANSLATE_BACK(v, min_, max_, size) \
    (((v) - (min_)) / ((max_) - (min_)) * (size))

#define TRANSLATE_BACK_INV(v, min_, max_, size) \
    ((size) - TRANSLATE_BACK((v), (min_), (max_), (size)))
    
// #define TRANSLATE_2D(id, size, x_min, x_max, y_min, y_max) \
//    (real2)((x_min) + (id).x*((x_max) - (x_min))/(size).x, (y_min) + (id).y*((y_max) - (y_min))/(size).y)

#define TRANSLATE_2D_INV_Y(id, size, x_min, x_max, y_min, y_max) \
    (real2)((x_min) + (id).x*((x_max) - (x_min))/(size).x, (y_min) + ((size).y - (id).y)*((y_max) - (y_min))/((size).y))

#define TRANSLATE_BACK_2D(v, x_min, x_max, y_min, y_max, size) \
    convert_int2_rtz( (real2) (((v).x - (x_min))/((x_max) - (x_min))*(size).x, \
                               ((v).y - (y_min))/((y_max) - (y_min))*(size).y ))
    
#define TRANSLATE_BACK_2D_INV_Y(v, x_min, x_max, y_min, y_max, size) \
    convert_int2_rtz( (real2) ( ((v).x - (x_min))/((x_max) - (x_min))*(size).x, \
                                (size).y - ((v).y - (y_min))/((y_max) - (y_min))*(size).y ))

#define NEAR(a, b, abs_error) (fabs((a) - (b)) < (abs_error))

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