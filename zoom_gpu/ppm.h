#ifndef PPMH
#define PPMH

typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

typedef struct {
	// real : x
	// imag : y
	double x, y;
} coord;


#endif
