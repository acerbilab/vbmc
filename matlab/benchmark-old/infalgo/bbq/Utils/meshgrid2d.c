#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
	int outrows, outcols, i, j;
	double *out, *in;

  if(nrhs != 2 || nlhs != 2)
    mexErrMsgTxt("Usage: [x,y] = meshgrid2d(X,Y");

	outrows = (mxGetM(prhs[1]) * mxGetN(prhs[1]));
	outcols = (mxGetM(prhs[0]) * mxGetN(prhs[0]));

	plhs[0] = mxCreateDoubleMatrix(outrows, outcols, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(outrows, outcols, mxREAL);

	out = mxGetPr(plhs[0]);
	in = mxGetPr(prhs[0]);

	for (i = 0; i < outrows; i++) 
		for (j = 0; j < outcols; j++) 
			out[i + outrows * j] = in[j];

	out = mxGetPr(plhs[1]);
	in = mxGetPr(prhs[1]);

	for (i = 0; i < outrows; i++) 
		for (j = 0; j < outcols; j++) 
			out[i + outrows * j] = in[i];

}
