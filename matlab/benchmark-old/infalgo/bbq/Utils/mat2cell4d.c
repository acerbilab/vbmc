#include "mex.h"

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  int rows,cols,files,ranks,rowstart,colstart,filestart,rankstart,
		xrows,xcols,xfiles,outrow,outcol,outfile,outrank,inrow,incol,
		infile,inrank;
  double *x,*rowsizes,*colsizes,*filesizes,*ranksizes;
	mwSize *indims, ndims, dims[4];

  if(nrhs != 5 || nlhs > 1)
    mexErrMsgTxt("Usage: c = mat2cell4d(x,m,n,l,o)");

	rows = mxGetM(prhs[1]) * mxGetN(prhs[1]);
	cols = mxGetM(prhs[2]) * mxGetN(prhs[2]);
	files = mxGetM(prhs[3]) * mxGetN(prhs[3]);
	ranks = mxGetM(prhs[4]) * mxGetN(prhs[4]);

	x = mxGetPr(prhs[0]);
	rowsizes = mxGetPr(prhs[1]);
	colsizes = mxGetPr(prhs[2]);
	filesizes = mxGetPr(prhs[3]);
	ranksizes = mxGetPr(prhs[4]);
	
	indims = mxGetDimensions(prhs[0]);
	xrows = indims[0];
	xcols = indims[1];
	xfiles = indims[2];

	ndims = 4;
	dims[0] = rows;
	dims[1] = cols;
	dims[2] = files;
	dims[3] = ranks;

  plhs[0] = mxCreateCellArray(ndims, dims);

	rowstart = 0;
	for (outrow = 0; outrow < rows; outrow++) {
		colstart = 0;
		for (outcol = 0; outcol < cols; outcol++) {
			filestart = 0;
			for (outfile = 0; outfile < files; outfile++) {
				rankstart = 0;
				for (outrank = 0; outrank < ranks; outrank++) {
					dims[0] = rowsizes[outrow];
					dims[1] = colsizes[outcol];
					dims[2] = filesizes[outfile];
					dims[3] = ranksizes[outrank];
					mxArray* temp = mxCreateNumericArray(ndims,
																							 dims,
																							 mxDOUBLE_CLASS,
																							 mxREAL);
					double* tempcontents = mxGetPr(temp);
					for (inrow = rowstart; 
							 inrow < rowstart + (int)(rowsizes[outrow]); 
							 inrow++)
						for (incol = colstart; 
								 incol < colstart + (int)(colsizes[outcol]); 
								 incol++)
							for (infile = filestart; 
									 infile < filestart + (int)(filesizes[outfile]); 
									 infile++) 
								for (inrank = rankstart; 
										 inrank < rankstart + (int)(ranksizes[outrank]); 
										 inrank++) {
									tempcontents[(inrow - rowstart) + 
															 ((int)(rowsizes[outrow]) * (incol - colstart)) +
															 ((int)(rowsizes[outrow]) * 
																(int)(colsizes[outcol]) * (infile - filestart)) +
															 ((int)(rowsizes[outrow]) * 
																(int)(colsizes[outcol]) * 
																(int)(filesizes[outfile]) * (inrank - rankstart))] = 
										x[inrow + (xrows * incol) + (xrows * xcols * infile) + 
											(xrows * xcols * xfiles * inrank)];
								}
					mxSetCell(plhs[0], 
										outrow + (rows * outcol) + (rows * cols * outfile) + 
										(rows * cols * files * outrank), 
										mxDuplicateArray(temp));	 
					mxDestroyArray(temp);
					rankstart += (int)(ranksizes[outrank]);
				}
				filestart += (int)(filesizes[outfile]);
			}
			colstart += (int)(colsizes[outcol]);
		}
		rowstart += (int)(rowsizes[outrow]);
	}
	
}
