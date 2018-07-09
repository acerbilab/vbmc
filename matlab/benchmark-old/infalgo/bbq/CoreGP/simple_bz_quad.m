function [mean, var] = simple_bz_quad(x, y, lower_bnd, upper_bnd, noises, input_scale, output_scale)
% returns the mean and variance for the integral of the function y(x) over
% a bounded region. The prior mean is assumed to be zero.
% _outputs_
% mean:         scalar mean of integral
% var:          scalar variance of integral
% _inputs_
% x:            locations of evaluations (n by 1 vector);
% y:            values of function y(x) (n by 1 vector);
% lower_bnd:    the lower bound specifying the domain.
% upper_bnd:    the upper bound specifying the domain.
% noises:       the vector of noises (n by 1 vector);
% input_scale:  input scale for gp
% output_scale: output scale for gp
% see O'Hagan `Some Bayesian Numerical Analysis' (1990ish).

Kfn = @(x1,x2) output_scale^2*normpdf(x1,x2,input_scale);

K = bsxfun(Kfn, x, x');
V = K + diag(noises);
V = improve_covariance_conditioning(V);
cholV = chol(V);

n = output_scale^2*(normcdf(upper_bnd, x, input_scale) - ...
                    normcdf(lower_bnd, x, input_scale));
                
mean = n'*(cholV\(cholV'\y));

c = output_scale^2*(...
    -2*input_scale/sqrt(2*pi)...
    +2*input_scale^2*normpdf(lower_bnd,upper_bnd,input_scale)...
    +(lower_bnd-upper_bnd)*erf((lower_bnd-upper_bnd)/(input_scale*sqrt(2)))...
    );

var = c - n'*(cholV\(cholV'\n));