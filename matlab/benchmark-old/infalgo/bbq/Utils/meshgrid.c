#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
	double *out, *in;

  if (nrhs == 2 && nlhs == 2) {
		int outrows, outcols, i, j;
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

	else if (nrhs == 3 && nlhs == 3) {
		int outrows, outcols, outrods,  i, j, k;
		outcols = (mxGetM(prhs[0]) * mxGetN(prhs[0]));
		outrows = (mxGetM(prhs[1]) * mxGetN(prhs[1]));
		outrods = (mxGetM(prhs[2]) * mxGetN(prhs[2]));

		mwSize dims[3];
		dims[0] = (mwSize) outrows;
		dims[1] = outcols;
		dims[2] = outrods;

		plhs[0] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
		plhs[1] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
		plhs[2] = mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);		

		out = mxGetPr(plhs[0]);
		in = mxGetPr(prhs[0]);

		for (i = 0; i < outrows; i++) 
			for (j = 0; j < outcols; j++) 
				for (k = 0; k < outrods; k++)
					out[i + outrows * j + outrows * outcols * k] = in[j];

		out = mxGetPr(plhs[1]);
		in = mxGetPr(prhs[1]);

		for (i = 0; i < outrows; i++) 
			for (j = 0; j < outcols; j++) 
				for (k = 0; k < outrods; k++)
					out[i + outrows * j + outrows * outcols * k] = in[i];

		out = mxGetPr(plhs[2]);
		in = mxGetPr(prhs[2]);

		for (i = 0; i < outrows; i++) 
			for (j = 0; j < outcols; j++) 
				for (k = 0; k < outrods; k++)
					out[i + outrows * j + outrows * outcols * k] = in[k];
		
	}
	else {
    mexErrMsgTxt("Usage: [x y] = meshgrid(X, Y) or [x y z] = meshgrid(X, Y, Z)");
	}

}
