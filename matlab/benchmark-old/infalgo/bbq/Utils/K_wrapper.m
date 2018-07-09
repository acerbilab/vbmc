function K = K_wrapper(covariance, hyper, xa, xb)

[dummy, K] = feval(covariance, hyper, xa, xb);