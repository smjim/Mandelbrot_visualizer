# Mandelbrot set 'rite of passage'
### a cuda-accelerated mandelbrot zoom video creation tool + cpu implementation of static visualizer

![mandelbrot set visualization](https://user-images.githubusercontent.com/78174712/174942522-cf79d6e3-fbee-4b69-b639-f72406418298.png)

![tmp0](https://user-images.githubusercontent.com/78174712/179056536-173e4d70-f0cf-467a-b35e-d32d437d6a72.png)

![tmp1](https://user-images.githubusercontent.com/78174712/179056545-8d8a023a-c172-4ecf-9980-2af609cf752f.png)
![tmp001](https://user-images.githubusercontent.com/78174712/179056551-07c5b955-7182-4e67-a3e9-76892f99a741.png)
![tmp2](https://user-images.githubusercontent.com/78174712/179056639-450f8b9e-c670-40aa-87fe-4f5d6ee70038.png)
![tmp0](https://user-images.githubusercontent.com/78174712/179056666-19341568-47a2-4626-8679-284a32e7cf39.png)
![tmp008](https://user-images.githubusercontent.com/78174712/179056701-587c9a90-a8dd-4e29-8ae7-d3a7fcb0969c.png)
![tmp042](https://user-images.githubusercontent.com/78174712/179056705-c4cdd353-9e58-4347-9d56-67eb49eef533.png)
![tmp054](https://user-images.githubusercontent.com/78174712/179056713-1ad713f2-73c8-4195-b6cc-0d837dd0bd4b.png)
![tmp075](https://user-images.githubusercontent.com/78174712/179056718-45233f9e-b8ee-408e-89eb-5c6fc7bff9b7.png)
![tmp117](https://user-images.githubusercontent.com/78174712/179056729-5441457f-a1b3-4e30-8b22-085ac83a625a.png)
![tmp134](https://user-images.githubusercontent.com/78174712/179056761-20cfc279-ec18-459a-a54d-37d16ae8fafc.png)
![tmp187](https://user-images.githubusercontent.com/78174712/179056768-1a28439e-7b25-4ea4-96d9-d209dec07363.png)
![tmp215](https://user-images.githubusercontent.com/78174712/179056774-1042b35c-7927-491c-9dfa-e735e552156f.png)

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
