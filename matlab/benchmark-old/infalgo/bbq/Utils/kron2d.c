#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  int leftrows, leftcols, rightrows, rightcols, rows, i, j, k, l;
	
  if(nrhs != 2 || nlhs > 1)
    mexErrMsgTxt("Usage: K = kron2d(X, Y)");

	leftrows = mxGetM(prhs[0]);
	leftcols = mxGetN(prhs[0]);
	rightrows = mxGetM(prhs[1]);
	rightcols = mxGetN(prhs[1]);

	rows = leftrows * rightrows;

  plhs[0] = mxCreateDoubleMatrix(rows, 
																 leftcols * rightcols, mxREAL);

	double* left = mxGetPr(prhs[0]);
	double* right = mxGetPr(prhs[1]);
	double* result = mxGetPr(plhs[0]);

	for (i = 0; i < leftrows; i++)
		for (j = 0; j < leftcols; j++) {
			double multiplier = left[j * leftrows + i];
			int rowoffset = i * rightrows;
			int coloffset = j * rightcols;
			for (k = 0; k < rightrows; k++)
				for (l = 0; l < rightcols; l++) {
					result[(coloffset + l) * rows + (rowoffset + k)] = 
						multiplier * right[l * rightrows + k];
				}
		}

	
}
