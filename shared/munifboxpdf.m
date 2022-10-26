function y = munifboxpdf(x,a,b)
%MUNIFBOXPDF Multivariate uniform box probability density function.
%   Y = MUNIFBOXPDF(X,A,B) returns the pdf of the multivariate uniform-box
%   distribution with bounds A and B, evaluated at the values in X. The 
%   multivariate uniform box pdf is the product of univariate uniform
%   pdfs in each dimension. 
%
%   For each dimension i, the univariate uniform-box pdf is defined as:
%
%                 |   
%                 |   ______________
%         p(X(i)) |   |            |
%                 |   |            |  
%                 |___|____________|_____
%                    A(i)          B(i)
%                           X(i)
%                          
%   X can be a matrix, where each row is a separate point and each column
%   is a different dimension. Similarly, A and B can also be matrices of 
%   the same size as X.
%
%   See also MUNIFBOXLOGPDF, MUNIFBOXRND.

% Luigi Acerbi 2022

y = exp(munifboxlogpdf(x,a,b));