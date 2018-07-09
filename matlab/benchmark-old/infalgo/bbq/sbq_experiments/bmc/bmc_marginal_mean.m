function m = bmc_marginal_mean( prior_mu_y, prior_sigma_y, Kinv, X, Y, f_vals, ...
                                quad_sigma_x, quad_sigma_y, quad_height, xstar )
% Computes the posterior marginal posterior mean of a GP,
% conditioned on points {X,Y}.
%
% Todo: replace quad_sigma_y with quad_sigma_y_given_x
%
% David Duvenaud
% March 2012
% =====================
[N,D] = size(xstar);

beta = Kinv * f_vals;

m = NaN(size(xstar));

for i = 1:N
    cur_xstar = xstar(i,:);
    m(i) = sum( beta .* quad_height ...
                  .* mvnpdf(X, cur_xstar, quad_sigma_x) ...
                  .* mvnpdf(Y, prior_mu_y, quad_sigma_y + prior_sigma_y ));
end
