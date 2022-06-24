CUDA Accelerated Mandelbrot zoom 
========================

This project utilizes parallel computing to improve the efficiency of Mandelbrot zooming.

This code produces a video zooming in a certain depth to a certain coordinate of the Mandelbrot set.

### How To Generate A Video:

1. Download and install [FFmpeg](https://ffmpeg.org/)

2. Choose an operation, compile and run it

	- zoom (default)
		* `make zoom`, then `./zoom`

3. Use FFmpeg to combine the output frames into a video:
```
ffmpeg -framerate 24 -i outfiles/tmp%03d.ppm -c:v libx264 -r 30 -pix_fmt yuv420p output_video.mp4
```
