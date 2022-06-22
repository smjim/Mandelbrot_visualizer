#include <iostream>
#include <fstream>

using namespace std;

void optimized(int max_iteration) {
	int res = 1000; // resolution of the set boundary
	
	ofstream data;
	data.open ("data.txt");
	for (int j = 0; j < res; j++) {
		for (int i = 0; i < res; i++) {
			float x0 = (i * 2.47/res) - 2.00; // Scaled to lie in Mandelbrot X scale (-2.00, 0.47)
			float y0 = (j * 2.24/res) - 1.12; // Scaled to lie in Mandelbrot Y scale (-1.12, 1.12)

			float x1 = 0;
			float y1 = 0;
			float x2 = 0;
			float y2 = 0;
			int iteration = 0;

			// z(n+1) = z(n)^2 + c
			while (x2 + y2 <= 4 && iteration < max_iteration) {
				y1 = 2*x1*y1 + y0;
				x1 = x2 - y2 + x0;

				x2 = x1*x1;
				y2 = y1*y1;
				iteration ++;
			}
			data << iteration << " ";
		}
		data << endl;
	}
}

void derbail(int max_iteration) {
	int res = 1000; // resolution of the set boundary
	
	ofstream data;
	data.open ("data.txt");
	for (int j = 0; j < res; j++) {
		for (int i = 0; i < res; i++) {
			float x0 = (i * 2.47/res) - 2.00; // Scaled to lie in Mandelbrot X scale (-2.00, 0.47)
			float y0 = (j * 2.24/res) - 1.12; // Scaled to lie in Mandelbrot Y scale (-1.12, 1.12)

			float x1 = 0;
			float y1 = 0;

			float dx = 1;
			float dy = 0;
			float dx_sum = 0;
			float dy_sum = 0;

			int iteration = 0;

			int dbail = 1e6;

			// z'(n+1) = 2 * z'(n) * z(n) + 1
			while (dx_sum*dx_sum + dy_sum*dy_sum < dbail && x1*x1 + y1*y1 <= 4 && iteration < max_iteration) {
				float xtmp = x1*x1 - y1*y1 + x0;
				y1 = 2*x1*y1 + y0;
				x1 = xtmp;

				float dxtmp = 2*(dx*x1 - dy*y1) + 1;
				dy = 2*(dy*x1 + dx*y1);
				dx = dxtmp;

				dx_sum += dx;
				dy_sum += dy;

				iteration ++;
			}
			data << iteration << " ";
		}
		data << endl;
	}
}

int main() {
	//optimized(20);
	derbail(20);

	return 0;
}
