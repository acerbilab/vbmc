function y = msmoothboxpdf(x,a,b,sigma)
%MSMOOTHBOXPDF Multivariate smooth-box probability density function.
%   Y = MSMOOTHBOXPDF(X,A,B,SIGMA) returns the pdf of the multivariate 
%   smooth-box distribution with pivots A and B and scale SIGMA, evaluated 
%   at the values in X. The multivariate smooth-box pdf is the product of 
%   univariate smooth-box pdfs in each dimension. 
%
%   For each dimension i, the univariate smooth-box pdf is defined as a
%   uniform distribution between pivots A(i), B(i) and Gaussian tails that
%   fall starting from p(A(i)) to the left (resp., p(B(i)) to the right) 
%   with standard deviation SIGMA(i).
%                          
%   X can be a matrix, where each row is a separate point and each column
%   is a different dimension. Similarly, A, B, and SIGMA can also be
%   matrices of the same size as X.
%
%   See also MSMOOTHBOXLOGPDF, MSMOOTHBOXRND.

% Luigi Acerbi 2022

y = exp(msmoothboxlogpdf(x,a,b,sigma));