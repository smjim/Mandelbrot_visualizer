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
- [COMPLETE] a new function needs to be defined that outputs coords b1, b2 given t
- [COMPLETE] linear interpolation zoom function needs to be non linear (slow down as approach)
	*log time
- [COMPLETE] instead of using boundaries to determine zoom, use logarithmically decreasing
radius centered on a point
- new coloring scheme is necessary
- [UNNECESSARY] eventually implement long double and extra long double to achieve higher zooms
- [COMPLETE] implement new addressing scheme independent of order (perform operations as if
in range -10, 10 and then afterwards apply order of magnitude (1-e210) for precision)
- [COMPLETE] implement new algorithm specially taylored for mandelbrot 'zoom' so that the gpu
doesnt have to deal with higher and higher dbail values for precision or iterations
	- [COMPLETE] peterbation theory series approximator is used by the kalles fraktaler for youtube zooms
- [COMPLETE] implement julia set explorer
- include configuration file to store data about zoom center, zoom depth, and julia set 
parametric equation *based off of make flags*
- progress bar to show when rendering complete
- operate on batches of frames in parallel to parallelize even further

## Errors:

- iterative process zooms in on the first iteration of the point instead of the point
	- (temporary fix) divide initial point by 2 in initialization
