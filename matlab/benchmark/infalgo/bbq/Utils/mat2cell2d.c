#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  int rows,cols,rowstart,colstart,xrows,i,j,k,l;
  double *x,*rowsizes,*colsizes;

  if(nrhs != 3 || nlhs > 1)
    mexErrMsgTxt("Usage: c = mat2cell2d(x,m,n)");

	rows = mxGetM(prhs[1]) * mxGetN(prhs[1]);
	cols = mxGetM(prhs[2]) * mxGetN(prhs[2]);

	x = mxGetPr(prhs[0]);
	rowsizes = mxGetPr(prhs[1]);
	colsizes = mxGetPr(prhs[2]);

	xrows = mxGetM(prhs[0]);
	
  plhs[0] = mxCreateCellMatrix(rows, cols);

	rowstart = 0;
	for (i = 0; i < rows; i++) {
		colstart = 0;
		for (j = 0; j < cols; j++) {
			mxArray* temp = mxCreateDoubleMatrix(rowsizes[i],
																					 colsizes[j],
																					 mxREAL);
			double* tempcontents = mxGetPr(temp);
			for (k = rowstart; k < rowstart + (int)(rowsizes[i]); k++)
				for (l = colstart; l < colstart + (int)(colsizes[j]); l++) {
					tempcontents[(k - rowstart) + 
											 ((int)(rowsizes[i]) * (l - colstart))] = 
						x[k + (xrows * l)];
				}
			mxSetCell(plhs[0], i + (rows * j), mxDuplicateArray(temp));	 
			mxDestroyArray(temp);
			colstart += (int)(colsizes[j]);
		}
		rowstart += (int)(rowsizes[i]);
	}
	
}
