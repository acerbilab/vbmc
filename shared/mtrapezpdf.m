function y = mtrapezpdf(x,a,u,v,b)
%MTRAPEZPDF Multivariate trapezoidal probability density function (pdf).
%   Y = MTRAPEZPDF(X,A,U,V,B) returns the pdf of the multivariate trapezoidal
%   distribution with external bounds A and B and internal points U and V,
%   evaluated at the values in X. The multivariate trapezoidal
%   pdf is the product of univariate trapezoidal pdfs in each dimension. 
%
%   For each dimension i, the univariate trapezoidal pdf is defined as:
%
%                 |       __________
%                 |      /|        |\
%         p(X(i)) |     / |        | \ 
%                 |    /  |        |  \
%                 |___/___|________|___\____
%                    A(i) U(i)     V(i) B(i)
%                             X(i)
%                          
%   X can be a matrix, where each row is a separate point and each column
%   is a different dimension. Similarly, A, B, C, and D can also be
%   matrices of the same size as X.
%
%   See also MTRAPEZLOGPDF, MTRAPEZRND.

% Luigi Acerbi 2022

y = exp(mtrapezlogpdf(x,a,u,v,b));