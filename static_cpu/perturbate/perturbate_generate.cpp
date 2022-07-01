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
// function based off of deep_zoom_point function in adelelopez/antelbrot
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

// finds iteration approximation for point e in relation to z using zn
// function based off of deep_zoom_point function in adelelopez/antelbrot
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
			int value = pert(e, zn);

			data << value << " ";
		}
		data << endl;
	}

	return 0;
}
