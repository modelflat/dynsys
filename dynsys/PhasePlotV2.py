import numpy
import pyopencl as cl

from .cl import ComputedImage, TypeConfig, generateCode, allocateImage

SOURCE = r"""

#define VARIABLE_TYPE2 float2
#define VARIABLE_TYPE3 float3
#define IMAGE_TYPE2 image2d_t
#define IMAGE_TYPE3 image3d_t

#define VAR2IMG_XY(point, bounds, image) \
    (VARIABLE_TYPE2)( \
            ((point).x - (bounds).s0) / ((bounds).s1 - (bounds).s0)*get_image_width (image), \
            get_image_height(image) - ((point).y - (bounds).s2) / ((bounds).s3 - (bounds).s2)*get_image_height(image) )

#define VAR2IMG_YZ(point, bounds, image) \
    (VARIABLE_TYPE2)( \
            ((point).y - (bounds).s2) / ((bounds).s3 - (bounds).s2)*get_image_width (image), \
            get_image_height(image) - ((point).z - (bounds).s4) / ((bounds).s5 - (bounds).s4)*get_image_height(image) )

#define VAR2IMG_XZ(point, bounds, image) \
    (VARIABLE_TYPE2)( \
            ((point).x - (bounds).s0) / ((bounds).s1 - (bounds).s0)*get_image_width (image), \
            get_image_height(image) - ((point).z - (bounds).s4) / ((bounds).s5 - (bounds).s4)*get_image_height(image) )

#define VAR2IMG_XYZ(point, bounds, image) \
    (VARIABLE_TYPE3)( \
            ((point).x - (bounds).s0) / ((bounds).s1 - (bounds).s0)*get_image_width (image), \
            ((point).y - (bounds).s2) / ((bounds).s3 - (bounds).s2)*get_image_height(image), \
            get_image_depth(image) - ((point).z - (bounds).s4) / ((bounds).s5 - (bounds).s4)*get_image_depth (image) )

#define VALID_COORD2(coord, image) \
    ((coord).x >= 0 && (coord).x < get_image_width (image) && \
     (coord).y >= 0 && (coord).y < get_image_height(image))  

#define VALID_COORD3(coord, image) \
    ((coord).x >= 0 && (coord).x < get_image_width (image) && \
     (coord).y >= 0 && (coord).y < get_image_height(image) && \
     (coord).z >= 0 && (coord).z < get_image_depth(image))


float4 colorForPoint(int maxIter, int skipIter, int iter, float3 point);
float4 colorForPoint(int maxIter, int skipIter, int iter, float3 point) {
#ifdef DYNAMIC_COLOR
    const float ratio = (float)(iter - skipIter) / (float)(maxIter - skipIter);
    return (float4)(hsv2rgb((float3)( 240.0 * (1.0 - ratio), 1.0, 1.0)), 1.0);
#else
    return DEFAULT_ENTITY_COLOR;
#endif
}

kernel void drawPhasePlot(
    const PARAMETERS_SIGNATURE,
    const BOUNDS bounds, 
    const int skip, const int iterations
#ifdef RENDER_XY
    , write_only IMAGE_TYPE2 resultXY
#endif
#ifdef RENDER_YZ
    , write_only IMAGE_TYPE2 resultYZ
#endif
#ifdef RENDER_XZ
    , write_only IMAGE_TYPE2 resultXZ
#endif
#ifdef RENDER_XYZ
    , write_only IMAGE_TYPE3 resultXYZ
#endif
) {
    const COORD_TYPE id = ID;
    VARIABLE_TYPE point = TRANSLATE_INV_Y(VARIABLE_TYPE, id, SIZE, bounds);
    float4 color;
    int2 coord2;
    int4 coord3;
    
    for (int i = 0; i < skip; ++i) {
        point = userFn(point, PARAMETERS);
    }
    
    for (int i = skip; i < iterations; ++i) {
        point = userFn(point, PARAMETERS);
        color = colorForPoint(iterations, skip, i, point);
#ifdef RENDER_XY
        coord2 = convert_int2_rtz(VAR2IMG_XY(point, bounds, resultXY));
        if (VALID_COORD2(coord2, resultXY)) {
            write_imagef(resultXY, coord2, color);
        }
#endif
#ifdef RENDER_YZ
        coord2 = convert_int2_rtz(VAR2IMG_YZ(point, bounds, resultYZ));
        if (VALID_COORD2(coord2, resultYZ)) {
            write_imagef(resultYZ, coord2, color);
        }
#endif
#ifdef RENDER_XZ
        coord2 = convert_int2_rtz(VAR2IMG_XZ(point, bounds, resultXZ));
        if (VALID_COORD2(coord2, resultXZ)) {
            write_imagef(resultXZ, coord2, color);
        }
#endif
#ifdef RENDER_XYZ
        coord3 = (int4)(convert_int3_rtz(VAR2IMG_XYZ(point, bounds, resultXYZ)), 0);
        if (VALID_COORD3(coord3, resultXYZ)) {
            write_imagef(resultXYZ, coord3.xzyw, color);
        }
#endif
    }
}
"""


class OutputConfig:

    def __init__(self, backColor2D=(1.0, 1.0, 1.0, 1.0), backColor3D=(0.0, 0.0, 0.0, 0.0),
                 shapeXY=None, shapeYZ=None, shapeXZ=None, shapeXYZ=None):
        self.shapes = []
        self.options = []
        self.colors = []
        if shapeXY is not None:
            self.shapes.append(shapeXY)
            self.options.append("-DRENDER_XY")
            self.colors.append(backColor2D)
        if shapeYZ is not None:
            self.shapes.append(shapeYZ)
            self.options.append("-DRENDER_YZ")
            self.colors.append(backColor2D)
        if shapeXZ is not None:
            self.shapes.append(shapeXZ)
            self.options.append("-DRENDER_XZ")
            self.colors.append(backColor2D)
        if shapeXYZ is not None:
            self.shapes.append(shapeXYZ)
            self.options.append("-DRENDER_XYZ")
            self.colors.append(backColor3D)

    def allocateImages(self, ctx, flags=cl.mem_flags.WRITE_ONLY):
        im = tuple(allocateImage(ctx, shape, flags) for shape in self.shapes)
        return tuple((img[0] for img in im)), tuple((img[1] for img in im))


def ceilToPow2(x):
    return 1 << int(numpy.ceil(numpy.log2(x)))


class PhasePlot(ComputedImage):

    def __init__(self,
                 ctx: cl.Context, queue: cl.CommandQueue,
                 gridShape: tuple,
                 spaceShape: tuple,
                 systemSource: str,
                 varCount: int,
                 paramCount: int,
                 typeConfig: TypeConfig,
                 outputConf: OutputConfig
                 ):
        super().__init__(ctx, queue, (1,)*varCount, spaceShape,
                         # sources
                         generateCode(typeConfig,
                                      boundsDims=varCount,
                                      variableCount=varCount,
                                      parameterCount=paramCount),
                         systemSource,
                         SOURCE,
                         #
                         typeConfig=typeConfig,
                         options=outputConf.options)
        self.gridShape = gridShape
        self.paramCount = paramCount
        self.outputConf = outputConf
        self.hostImages, self.devImages = outputConf.allocateImages(self.ctx)

    def clearImages(self):
        for img, shp, clr in zip(self.devImages, self.outputConf.shapes, self.outputConf.colors):
            cl.enqueue_fill_image(
                self.queue, img,
                color=numpy.array(clr, dtype=numpy.float32),
                origin=(0,)*len(shp), region=shp
            )

    def readImages(self):
        for himg, dimg, shp in zip(self.hostImages, self.devImages, self.outputConf.shapes):
            cl.enqueue_copy(
                self.queue, himg, dimg,
                origin=(0,)*len(shp), region=shp
            )

    def __call__(self, parameters, iterations, skip=0):
        self.clearImages()

        space = tuple(self.spaceShape[i] if i < len(self.spaceShape) else numpy.nan
                      for i in range(ceilToPow2(len(self.spaceShape))))

        self.program.drawPhasePlot(
            self.queue, self.gridShape, None,
            *self.wrapArgs(self.paramCount, *parameters),
            numpy.array(space, dtype=self.typeConf.boundsType),
            numpy.int32(skip), numpy.int32(iterations),
            *self.devImages
        )

        self.readImages()

        return self.hostImages
