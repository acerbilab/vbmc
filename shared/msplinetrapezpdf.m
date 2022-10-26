function y = msplinetrapezpdf(x,a,b,c,d)
%MSPLINETRAPEZPDF Multivariate spline-trapezoidal probability density fcn (pdf).
%   Y = MSPLINETRAPEZPDF(X,A,B,C,D) returns the pdf of the multivariate 
%   spline-trapezoidal distribution with external bounds A and D and internal 
%   points B and C, evaluated at the values in X. The multivariate pdf is 
%   the product of univariate spline-trapezoidal pdfs in each dimension. 
%
%   For each dimension i, the univariate spline-trapezoidal pdf is defined 
%   as a trapezoidal pdf whose points A, B and C, D are connected by cubic
%   splines such that the pdf is continuous and its derivatives at A, B, C,
%   and D are zero (so the derivatives are also continuous):
%
%                 |       __________
%                 |      /|        |\
%         p(X(i)) |     / |        | \ 
%                 |    /  |        |  \
%                 |___/___|________|___\____
%                    A(i) B(i)     C(i) D(i)
%                             X(i)
%                          
%   X can be a matrix, where each row is a separate point and each column
%   is a different dimension. Similarly, A, B, C, and D can also be
%   matrices of the same size as X.
%
%   See also MSPLINETRAPEZLOGPDF, MSPLINETRAPEZRND.

% Luigi Acerbi 2022

y = exp(msplinetrapezlogpdf(x,a,b,c,d));