function [y,dy] = testpdf(x)

D = numel(x);
sigma = 1:D;
y = -0.5*sum(x.^2./sigma.^2);
dy = -x./sigma.^2;