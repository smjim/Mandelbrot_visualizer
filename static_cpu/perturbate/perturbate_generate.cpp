#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <gmpxx.h>	// for bigNum calculations

using std::cout;
using std::endl;
using std::vector;

struct coord {
	double x;	// real component
	double y;	// imaginary component
};

// high precision point used for perturbation theory method
// produces a list of iteration values used to compute the surrounding points
// function based off of deez_zoom_point function in adelelopez/antelbrot
vector<coord> gen_zn(const mpf_class &center_r, const mpf_class &center_i, 
				int depth)
{
    vector<coord> v;
    mpf_class xn_r = center_r;
    mpf_class xn_i = center_i;

    for (int i = 0; i != depth; ++i)
    {   
        // pre multiply by two
        mpf_class re = xn_r + xn_r;
        mpf_class im = xn_i + xn_i;

        coord c = {re.get_d(), im.get_d()};

        v.push_back(c);

        // make sure our numbers don't get too big
        if (re > 1024 || im > 1024 || re < -1024 || im < -1024)
            return v;

        // calculate next iteration, remember re = 2 * xn_r
        xn_r = xn_r * xn_r - xn_i * xn_i + center_r;
        xn_i = re * xn_i + center_i;
    }   
    return v;
}

/*
void gen_zn(coord &z, vector<coord> &zn, int max_iteration) {
	// using high-precision operations, determine zn for all n = 1 .. max_iteration and store in array	
	// uses optimized naive method

	coord z1 = {0, 0};
	coord z2 = {0, 0};
	int iteration = 0;
	
	// z(n+1) = z(n)^2 + c
	while (iteration < max_iteration) {
		z1.y = 2*z1.x*z1.y + z.y; 
		z1.x = z2.x - z2.y + z.x; 
	
		z2.x = z1.x*z1.x;
		z2.y = z1.y*z1.y;

		zn[iteration] = z1;
		iteration ++; 

		if (z2.x + z2.y > 4) {
			cout << "given point not part of mandelbrot set: Z = (" << z.x << ", " << z.y << ")" << endl;
			abort();
		}
	}
}
*/

int pert(coord &e, vector<coord> &zn) {

	int iter = 0;
	int max_iter = zn.size();

	double en_size;

	// run the iteration loop
	coord dn = e;	//d0 = e
	do
	{
		// dn *= zn[iter] + dn;
		double tmp = (dn.x*zn[iter].x) - (dn.y*zn[iter].y) + (dn.x*dn.x) - (dn.y*dn.y);
		dn.y = (dn.x*zn[iter].y) + (dn.y*zn[iter].x) + 2.*(dn.x*dn.y);
		dn.x = tmp;

		// dn += e;
		dn = {dn.x + e.x, dn.y + e.y};

		iter ++;

		// en_size = norm(0.5*zn[iter] + dn);
		en_size = sqrt(pow(0.5*zn[iter].x + dn.x, 2) + pow(0.5*zn[iter].y + dn.y, 2)); 
	}
	while (en_size < 256 && iter < max_iter);
	return iter;
}

/*
{
    int window_radius = (size.x < size.y) = size.x : size.y;
    // find the complex number at the center of this pixel
    std::complex<double> d0 (radius * (2 * i - (int) size.x) / window_radius,
                             -radius * (2 * j - (int) size.y) / window_radius);

    int iter = 0;

    int max_iter = x.size();

    double zn_size;
    // run the iteration loop
    std::complex<double> dn = d0; 
    do  
    {   
        dn *= x[iter] + dn; 
        dn += d0; 
        ++iter;
        zn_size = std::norm(x[iter] * 0.5 + dn);

        // use bailout radius of 256 for smooth coloring.
    }   
    while (zn_size < 256 && iter < max_iter);
	return iter;
}
*/

float perturbate(vector<coord> zn, coord sigma, coord e) {

	int max_iteration = zn.size();

	// iteration 0
	coord a = {1, 0}; 
	coord b = {0, 0}; 
	coord c = {0, 0}; 
	coord dn;	// d0 = sigma
	coord en;	// e0 = e

	// find dn given zn, sigma 
	for (int n = 0; n < max_iteration; n++) {

		// find dn given d0 (sigma), zn

		coord sigma_2 = {sigma.x*sigma.x - sigma.y*sigma.y      , 2*sigma.x*sigma.y};						// sigma^2
		coord sigma_3 = {sigma_2.x*sigma.x - sigma_2.y*sigma.y  , sigma_2.x*sigma.y + sigma_2.y*sigma.x};	// sigma^3
	
		// d(n) = a(n)*sigma + b(n)*sigma^2 + c(n)*sigma^3 + o(sigma^4) 
		dn.x = (a.x*sigma.x - a.y*sigma.y) + (b.x*sigma_2.x - b.y*sigma_2.y) + (c.x*sigma_3.x - c.y*sigma_3.y);
		dn.y = (a.x*sigma.y + a.y*sigma.x) + (b.x*sigma_2.y + b.y*sigma_2.x) + (c.x*sigma_3.y + c.y*sigma_3.x);
	
		c = {2*(zn[n].x*c.x - zn[n].y*c.y) + 2*(a.x*b.x - a.y*b.y)  , 2*(zn[n].x*c.y + zn[n].y*c.x) + 2*(a.x*b.y + a.y*b.x)};   // c(n+1) = 2zn(n)c(n) + 2a(n)b(n)
		b = {2*(zn[n].x*b.x - zn[n].y*b.y) + (a.x*a.x - a.y*a.y)    , 2*(zn[n].x*b.y + zn[n].y*b.x) + 2*(a.x*a.y)};             // b(n+1) = 2zn(n)b(n) + a(n)a(n)
		a = {2*(zn[n].x*a.x - zn[n].y*a.y) + 1                      , 2*(zn[n].x*a.y + zn[n].y*a.x)};                           // a(n+1) = 2zn(n)a(n) + 1


		// using dn, e, approximate en
		// e = z + d
		en = {zn[n].x + dn.x, zn[n].y + dn.y};

		// if magnitude of en > 2, return n; otherwise loop until max_iteration
		float mag = sqrt(pow(en.x, 2) + pow(en.y,2));
		if (mag > 2.0 || n == max_iteration-1) return n; //mag;
	}
	return -1;
}

int main() {
	// parameters
	int max_iteration = 300;
	int res = 200;				// resolution * resolution = # of points (e)
	coord b1 = {-2.20, -1.12};	// bounding coord 1
	coord b2 = { 0.48,  1.12};	// bounding coord 2
	//coord b1 = {-0.7250, -0.3576};	// bounding coord 1
	//coord b2 = {-0.7248, -0.3574};	// bounding coord 2

	// the "base point" from which n will be determined for all other points (e)
	// this point is also the center of the zoom in zooming versions (N/A to static)
	//coord z = {-1.7499984109937408, -0.0000000000000016}; 
	//coord z = {-0.724973, -0.357569};
	coord z ={0, 0};

	mpf_class zx(z.x, 100);	
	mpf_class zy(z.y, 100);	

	vector<coord> zn(max_iteration);
	zn = gen_zn(zx, zy, max_iteration);

	// instead of generating distance, sigma, and z(n) values for every point, operate on one point at a time
	// for all points between b1, b2
	std::ofstream data;
	data.open ("perturbate_data.txt");
	for (int j = 0; j < res; j++) {
		for (int i = 0; i < res; i++) {
			// point to be operated upon
			coord e;
			e.x = (i * (b2.x - b1.x)/res) + b1.x; // Scaled to lie in X scale (b1.re, b2.re)
			e.y = (j * (b2.y - b1.y)/res) + b1.y; // Scaled to lie in Y scale (b1.im, b2.im)

			// find distance vector (sigma)
			coord sigma = {e.x - z.x, e.y - z.y};


			/*---------------- perturbate function explanation:
				iterate through n = 1, max_iteration
					find dn given zn, sigma
					approximate en given dn, e
					if magnitude(en) > 2, return n
			 	if n == max_iteration, return max_iteration
			*/
			//float value = perturbate(zn, sigma, e);
			int value = pert(e, zn);

			data << value << " ";
		}
		data << endl;
	}

	return 0;
}

/*	NOTES	

	sigma[] = d0[] = list of values of distance from point to target point (z)
	diff[] = list of values for escape time of points (paired with sigma[] list)
	
	* possibly try out making e vector into two mpf_class values to see if it increases precision, will greatly increase time because inside loop
*/
