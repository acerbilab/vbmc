function [ x ] = inv_logistic( f, lower, upper)
% logistic function

x = -log(bsxfun(@rdivide, upper - lower, f - lower) - 1);


