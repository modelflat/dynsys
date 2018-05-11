COMMON_SOURCE = """

#define ID_2D (int2)(get_global_id(0), get_global_id(1))
#define ID_2D_Y_INV (int2)(get_global_id(0), get_global_size(1) - get_global_id(1))

#define SIZE_2D (int2)(get_global_size(0), get_global_size(1))

#define TRANSLATE(id, size, x_min, x_max, y_min, y_max) \
    (real2)((x_min) + (id).x*((x_max) - (x_min))/(size).x, (y_min) + (id).y*((y_max) - (y_min))/(size).y)

#define TRANSLATE_INV_Y(id, size, x_min, x_max, y_min, y_max) \
    (real2)((x_min) + (id).x*((x_max) - (x_min))/(size).x, (y_min) + ((size).y - (id).y)*((y_max) - (y_min))/((size).y))

#define TRANSLATE_BACK      (v, size, x_min, x_max, y_min, y_max) \
    convert_int2_rtz( (real2) (((v).x - (x_min))/((x_max) - (x_min))*(size).x, \
                               ((v).y - (y_min))/((y_max) - (y_min))*(size).y ))
    
#define TRANSLATE_BACK_INV_Y(v, size, x_min, x_max, y_min, y_max) \
    convert_int2_rtz( (real2) ( ((v).x - (x_min))/((x_max) - (x_min))*(size).x, \
                                (size).y - ((v).y - (y_min))/((y_max) - (y_min))*(size).y ))

#define NEAR(a, b, abs_error) (fabs((a) - (b)) < (abs_error))

"""