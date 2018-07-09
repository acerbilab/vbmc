function vars = bmc_marginal_variance( prior_mu_y, prior_sigma_y, Kinv, X, Y, ...
                                      quad_sigma_x, quad_sigma_y, quad_height, xstar )
% Computes the posterior marginal variance of a GP,
% conditioned on points {X,Y}.
%
% % Todo: replace quad_sigma_y with quad_sigma_y_given_x
%
% David Duvenaud
% March 2012
% =====================
[N,D] = size(xstar);

vars = NaN(N,1);

% Note: Bringing this term outside the for loop assumes that the kernel is
% stationary.
% Original, more general form:
% t1 = quad_height * mvnpdf(cur_xstar, cur_xstar, quad_sigma_x) ...
%                  * mvnpdf(0,0, quad_sigma_y + 2.* prior_sigma_y );

t1 = quad_height * mvnpdf(0, 0, quad_sigma_x) ...
                 * mvnpdf(0, 0, quad_sigma_y + 2.* prior_sigma_y );

for i = 1:N
    cur_xstar = xstar(i,:);

    z = quad_height .* mvnpdf(X, cur_xstar, quad_sigma_x) ...
                    .* mvnpdf(Y, prior_mu_y, quad_sigma_y + prior_sigma_y );
    vars(i) = t1 - z' * Kinv * z;
end
