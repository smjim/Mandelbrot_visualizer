#include <iostream>

using namespace std;

class Complex {
	public:
		float real, imag;
		bool belongs = false;
		Complex(float r = 0, float i = 0) {real = r;	imag = i;}
		 
		// This is automatically called when '+' is used with
		// between two Complex objects
		Complex operator + (Complex const &obj) {
			Complex res;
			res.real = real + obj.real;
			res.imag = imag + obj.imag;
			return res;
		}

		void square() {
			real = real*real - imag*imag;
			imag = 2*real*imag;
		}

		void testPoint(int iterations) {
			// z_n+1 = (z_n)^2 + c | z_0 = 0
			Complex z(0, 0);
			for (int i = 0; i < iterations; i++) {
				z.square();
				z.real += real;
				z.imag += imag;
			}
			if (z.real < 100) {		// function used to determine if point belongs
				belongs = true;
			}
		}

		void print() { cout << real << " + i" << imag << ": "; }
};

int main() {
	int res = 100; // resolution of the set boundary
	cout << "[";
	for (int j = 0; j < res; j++) {
		cout << "[";
		for (int i = 0; i < res; i++) {
			float x = (i * 2.47/res) - 2.00; // Scaled to lie in Mandelbrot X scale (-2.00, 0.47)
			float y = (j * 2.24/res) - 1.12; // Scaled to lie in Mandelbrot Y scale (-1.12, 1.12)

			Complex point(x, y);
			point.testPoint(50);
			//point.print();
			cout << point.belongs << ",";
		}
		cout << "], ";
	}
	cout << "]\n";
}
