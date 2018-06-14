#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[],
								 int nrhs, const mxArray *prhs[])
{
	int inrows, incols, outrows, outcols, i, j, k, l, rowoff, coloff;
	double* out;
	mxArray* temp;
	
  if(nrhs != 1 || nlhs > 1)
    mexErrMsgTxt("Usage: m = mat2cell2d(c)");
	
	inrows = mxGetM(prhs[0]);
	incols = mxGetN(prhs[0]);
	
	outrows = 0;
	for (i = 0; i < inrows; i++)
		outrows += mxGetM(mxGetCell(prhs[0], i));
	
	outcols = 0;
	for (i = 0; i < incols; i++)
		outcols += mxGetN(mxGetCell(prhs[0], i * inrows));
	
	plhs[0] = mxCreateDoubleMatrix(outrows, outcols, mxREAL);
	out = mxGetPr(plhs[0]);

	rowoff = 0;
	for (i = 0; i < inrows; i++) 
	{
		coloff = 0;
		for (j = 0; j < incols; j++) 
    {
			temp = mxGetCell(prhs[0], i + inrows * j);
			double* tempin = mxGetPr(temp);
			for (k = 0; k < mxGetM(temp); k++) 
				for (l = 0; l < mxGetN(temp); l++) 
        {
					out[rowoff + k + (outrows * (coloff + l))] = 
						tempin[k + mxGetM(temp) * l];
				}
			coloff += mxGetN(temp);
		}
		rowoff += mxGetM(temp);
	}
			
}
