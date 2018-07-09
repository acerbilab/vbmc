#include <math.h>
#include "mex.h"

/* function h = toepsolve(r,q);
 * TOEPSOLVE Solve Toeplitz system of equations.
 * Solves R*h = q, where R is the symmetric Toeplitz matrix
 * whos first column is r
 * Assumes all inputs are real
 * Inputs:
 * r - first column of Toeplitz matrix, length n
 * q - rhs vector, length n
 * Outputs:
 * h - length n solution
 *
 * Algorithm from Roberts & Mullis, p.233
 *
 * Author: T. Krauss, Sept 10, 1997
 */
void mexFunction(
    int nlhs,
    mxArray *plhs[],
    int nrhs,
    const mxArray *prhs[]
)
{
   double *a,*h,beta;
   int j,k;

   double eps = mxGetEps();
   int n = (mxGetN(prhs[0])>=mxGetM(prhs[0])) ? mxGetN(prhs[0]) :
mxGetM(prhs[0]) ;
   double *r = mxGetPr(prhs[0]);
   double *q = mxGetPr(prhs[1]);
   double alpha = r[0];

   n = n - 1;

   plhs[0] = mxCreateDoubleMatrix(n+1,1,0);
   h = mxGetPr(plhs[0]);

   h[0] = q[0]/r[0];

   a = mxCalloc((n+1)*(n+1),sizeof(double));
   if (a == NULL) {
       mexErrMsgTxt("Sorry, failed to allocate buffer.");
   }

   a[(0*(n+1))+0] = 1.0;

   for (k = 1; k <= n; k++) {
       a[(k*(n+1))+k-1] = 0;
       a[(0*(n+1))+k] = 1.0;
       beta = 0.0;
       for (j = 0; j <= k-1; j++) {
           beta += r[k-j]*a[(j*(n+1))+k-1];
       }
       beta /= alpha;
       for (j = 1; j <= k; j++) {
           a[(j*(n+1))+k] = a[(j*(n+1))+k-1] - beta*a[((k-j)*(n+1))+k-1];
       }
       alpha *= (1 - beta*beta);
       h[k] = q[k];
       for (j = 0; j <= k-1; j++) {
           h[k] -= r[k-j]*h[j];
       }
       h[k] /= alpha;
       for (j = 0; j <= k-1; j++) {
           h[j] += a[((k-j)*(n+1))+k]*h[k];
       }
   } /* loop over k */

   mxFree(a);

   return;
}


