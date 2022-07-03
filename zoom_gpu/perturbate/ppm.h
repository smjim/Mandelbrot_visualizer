#ifndef PPMH
#define PPMH

struct PPMPixel {
     unsigned char red,green,blue;
};

struct PPMImage {
     int x, y;
     PPMPixel *data;
};

struct coord {
	// real : x
	// imag : y
	double x, y;
};


#endif
