#ifndef PRECISION_INCLUDED
#define PRECISION_INCLUDED

#include <vector>  
#include <gmpxx.h>
#include "ppm.h"

using std::vector;

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

#endif // PRECISION_INCLUDED 
