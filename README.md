# Mandelbrot set 'rite of passage'
### making a cuda-accelerated mandelbrot zoom video creation tool

![mandelbrot set visualization](https://user-images.githubusercontent.com/78174712/174942522-cf79d6e3-fbee-4b69-b639-f72406418298.png)

## Creating an output
	- CPU static image:
		1. `cd static_cpu`
		2. `g++ generate_set.cpp -o Generate`
		3. `./Generate`
		4. `python3 display.py`
	- GPU dynamic video:
		1. edit boundaries() function in ppm.cu
		2. `make`
		3. `./zoom`
		4. `ffmpeg -framerate 24 -i outfiles/tmp%03d.ppm -c:v libx264 -r 30 -pix_fmt yuv420p output_video.mp4`

## Design and implementation
GPU accelerated video creation tool developed using code adapted from [cuda-video-project](https://github.com/bojdell/cuda-video-project)
Each thread calculates its complex coordinates given two boundary points + width and height of final image, then calculates the color of the pixel using an optimized mandelbrot set-checking method from wikipedia.
Boundary points passed to the kernel are given by a function with input t (frame #), which in turn determines the level of zoom achieved.
