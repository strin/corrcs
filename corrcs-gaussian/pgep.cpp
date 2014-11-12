#include "stdio.h"
#include "src/PolyaGamma.h"
#include "mex.h"

PolyaGamma pg;
RNG rng;

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	double n = *mxGetPr(prhs[0]), z = *mxGetPr(prhs[1]), iter = *mxGetPr(prhs[2]);
	if(nlhs == 0) return;
	plhs[0] = mxCreateDoubleMatrix(1,1, mxREAL);
	double* res = mxGetPr(plhs[0]); 
	res[0] = 0;
	for(int i = 0; i < iter; i++) 
		res[0] += pg.draw(n, z, rng);
	res[0] /= iter;
}

int main() {

}