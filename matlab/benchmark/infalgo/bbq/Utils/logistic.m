function [ f ] = logistic( x, lower, upper)
% logistic function, bounded by lower and upper

f = lower + bsxfun(@rdivide,upper-lower,(1+exp(-x)));


