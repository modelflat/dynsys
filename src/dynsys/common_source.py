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

// #define TRANSLATE_BACK_2D(v, x_min, x_max, y_min, y_max, size) \
//     convert_int2_rtz( (real2) (((v).x - (x_min))/((x_max) - (x_min))*(size).x, \
//                                ((v).y - (y_min))/((y_max) - (y_min))*(size).y ))
    
#define TRANSLATE_BACK_2D_INV_Y(v, x_min, x_max, y_min, y_max, size) \
    convert_int2_rtz( (real2) ( ((v).x - (x_min))/((x_max) - (x_min))*(size).x, \
                                (size).y - ((v).y - (y_min))/((y_max) - (y_min))*(size).y ))

#define NEAR(a, b, abs_error) (fabs((a) - (b)) < (abs_error))

"""