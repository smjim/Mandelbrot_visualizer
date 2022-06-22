#include <iostream>
#include <fstream>

using namespace std;

class Complex {
	public:
		float real, imag;
		int belongs = 0;
		Complex(float r = 0, float i = 0) {real = r;	imag = i;}
		 
		Complex operator + (Complex const &obj) {
			Complex res;
			res.real = real + obj.real;
			res.imag = imag + obj.imag;
			return res;
		}

		void square() {
			float tmp = real;
			real = real*real - imag*imag;
			imag = 2*tmp*imag;
		}

		void testPoint(int max_iterations) {
			// z_n+1 = (z_n)^2 + c | z_0 = 0
			Complex z(0, 0);

			// breakout condition is that magnitude of point is <= 2
			while (z.real*z.real + z.imag*z.imag <= 4. && belongs < max_iterations) {	
				z.square();
				z.real += real;
				z.imag += imag;
				belongs ++;
			}
		}

		void print() { cout << real << " + i" << imag << ": "; }
};

void naive() {
	int res = 1000; // resolution of the set boundary
	
	ofstream data;
	data.open ("data.txt");
	for (int j = 0; j < res; j++) {
		for (int i = 0; i < res; i++) {
			float x = (i * 2.47/res) - 2.00; // Scaled to lie in Mandelbrot X scale (-2.00, 0.47)
			float y = (j * 2.24/res) - 1.12; // Scaled to lie in Mandelbrot Y scale (-1.12, 1.12)

			Complex point(x, y);
			point.testPoint(100);
			//point.print();
			data << point.belongs << " ";
		}
		data << endl;
	}
}

int main() {
	naive();

	return 0;
}
