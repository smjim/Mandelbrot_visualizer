# TODO list for creating functioning code

- [COMPLETE] actual code for generating mandelbrot set will be placed in ppmKernel.cu
	- the kernel operates once per video frame, make it so that instead of 
	operating upon a video frame to get each pixel value, operate upon a
	specified state to get each pixel color:

	GPU Kernel Code (given boundary1, boundary2):

	__global__ kernel(b1, b2) {
		// each thread calculates complex coords of its pixel given b1, b2
		// each thread performs operation to determine if its pixel belongs

		// each thread colors its pixel accordingly
	}
	- pass in float b1, b2 instead of image frame to function

- [COMPLETE] right now, the image is bounded by the width and height of the original
video file, which is not wanted. a fix would involve removing all refs to
the original video file, and passing in new width and height
- a new function needs to be defined that outputs coords b1, b2 given t
- new coloring scheme is necessary
- eventually implement long double and extra long double to achieve higher zooms
