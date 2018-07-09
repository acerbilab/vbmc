#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  int rows,cols,files,rowstart,colstart,filestart,xrows,xcols,
		i,j,k,l,m,n;
  double *x,*rowsizes,*colsizes,*filesizes;
	mwSize *indims, ndims, dims[3];
	mwIndex index;

  if(nrhs != 4 || nlhs > 1)
    mexErrMsgTxt("Usage: c = mat2cell3d(x,m,n,l)");

	rows = mxGetM(prhs[1]) * mxGetN(prhs[1]);
	cols = mxGetM(prhs[2]) * mxGetN(prhs[2]);
	files = mxGetM(prhs[3]) * mxGetN(prhs[3]);

	x = mxGetPr(prhs[0]);
	rowsizes = mxGetPr(prhs[1]);
	colsizes = mxGetPr(prhs[2]);
	filesizes = mxGetPr(prhs[3]);
	
	indims = mxGetDimensions(prhs[0]);
	xrows = indims[0];
	xcols = indims[1];

	ndims = 3;
	dims[0] = rows;
	dims[1] = cols;
	dims[2] = files;

  plhs[0] = mxCreateCellArray(ndims, dims);

	rowstart = 0;
	for (i = 0; i < rows; i++) {
		colstart = 0;
		for (j = 0; j < cols; j++) {
			filestart = 0;
			for (k = 0; k < files; k++) {
				dims[0] = rowsizes[i];
				dims[1] = colsizes[j];
				dims[2] = filesizes[k];
				mxArray* temp = mxCreateNumericArray(ndims,
																						 dims,
																						 mxDOUBLE_CLASS,
																						 mxREAL);
				double* tempcontents = mxGetPr(temp);
				for (l = rowstart; l < rowstart + (int)(rowsizes[i]); l++)
					for (m = colstart; m < colstart + (int)(colsizes[j]); m++)
						for (n = filestart; n < filestart + (int)(filesizes[k]); n++) {
							tempcontents[(l - rowstart) + 
													 ((int)(rowsizes[i]) * (m - colstart)) +
													 ((int)(rowsizes[i]) * 
														(int)(colsizes[j]) * (n - filestart))] = 
								x[l + (xrows * m) + (xrows * xcols * n)];
						}
				mxSetCell(plhs[0], 
									i + (rows * j) + (rows * cols * k), 
									mxDuplicateArray(temp));	 
				mxDestroyArray(temp);
				filestart += (int)(filesizes[k]);
			}
			colstart += (int)(colsizes[j]);
		}
		rowstart += (int)(rowsizes[i]);
	}
	
}

