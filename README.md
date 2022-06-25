# Mandelbrot set 'rite of passage'
### a cuda-accelerated mandelbrot zoom video creation tool + cpu implementation of static visualizer

![mandelbrot set visualization](https://user-images.githubusercontent.com/78174712/174942522-cf79d6e3-fbee-4b69-b639-f72406418298.png)


https://user-images.githubusercontent.com/78174712/175756549-c3dfdf9c-4031-4ae5-a88e-972d548550ad.mp4
(seahorse valley zoom)


## Creating an output
	- CPU static image:
		1. Go to cpu directory: `cd static_cpu`
		2. Generate data: `g++ generate_set.cpp -o Generate`, and `./Generate`
		3. Display data: `python3 display.py`
	- GPU dynamic video:
		1. Edit center, MAX_R, MIN_R, and length in ppm.cu
		2. Compile and generate the video: `make`, and `./zoom`
		3. Combine output frames: `ffmpeg -framerate 24 -i outfiles/tmp%03d.ppm -c:v libx264 -r 30 -pix_fmt yuv420p output_video.mp4`

## Design and implementation
GPU accelerated video creation tool developed using code adapted from [cuda-video-project](https://github.com/bojdell/cuda-video-project).
Each thread calculates its complex coordinates given two boundary points + width and height of final image, then calculates the color of the pixel using an optimized mandelbrot set-checking method from wikipedia.
Boundary points passed to the kernel are given by a function with input t (frame #), which in turn determines the level of zoom achieved with each frame.
