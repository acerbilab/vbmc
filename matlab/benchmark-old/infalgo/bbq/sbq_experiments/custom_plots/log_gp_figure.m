% Bare-bones Demo of Bayesian Quadrature with Log-GPs
%
% Written to help understand Mike Osborne's AISTATS paper.
%
% Takes a function that's better modeled in the log-domain and tries to
% integrate under it.
%
% David Duvenaud
% Jan 2012
% ===========================


% Define a likelihood function (one that is well-modeled by a gp on the log).
loglikelihood = @(x)(normpdf(x,1,1).*2+normpdf(x,3,1).*30);
likelihood = @(x)exp(loglikelihood(x));

% Plot likelihood.
N = 200;
xrange = linspace( -2, 6, N )';
figure(11);
like_handle = plot( xrange, likelihood(xrange), 'k'); hold on; % pause;

% Choose function sample points.
n_f_samples = 4;
function_sample_points = linspace( 0, 4, n_f_samples);

% Evaluate likelihood and log-likelood at sample points.
y = likelihood(function_sample_points)';
logy = loglikelihood(function_sample_points)';
sp_handle = plot( function_sample_points, y, 'kd', 'LineWidth', 2); hold on;
%pause;


% Model likelihood with a GP.
% =================================

% Define quadrature hypers.
quad_length_scale = 1;
quad_kernel = @(x,y)exp( - 0.5 * ( ( x - y ) .^ 2 ) ./ quad_length_scale );
quad_noise = 1e-6;

% Perform GP inference to get posterior mean function.
K = bsxfun(quad_kernel, function_sample_points', function_sample_points ); % Fill in gram matrix
C = inv( K + quad_noise^2 .* diag(N) ); % Compute inverse covariance
weights = C * y;  % Now compute kernel function weights.
posterior = @(x)(bsxfun(quad_kernel, function_sample_points, x) * weights); % Construct posterior function.

% Plot likelihood-GP posterior.
gpf_handle = plot( xrange, posterior(xrange), 'r'); hold on;
bad_gp_estimate = mean(posterior(xrange))
%pause;


% Model log-likelihood with a GP
% =================================
quad_log_length_scale = 1;
quad_log_kernel = @(x,y)exp( - 0.5 * (( x - y ) .^ 2 ) ./ quad_log_length_scale );
quad_log_noise = 1e-6;

K = bsxfun(quad_log_kernel, function_sample_points', function_sample_points ); % Fill in gram matrix
C = inv( K + quad_log_noise^2 .* diag(N) ); % Compute inverse covariance
weights = C * logy;  % Now compute kernel function weights.
log_posterior = @(x)(bsxfun(quad_log_kernel, function_sample_points, x) * weights); % Construct posterior function.
log_posterior_variance = @(x)(bsxfun(quad_log_kernel, x, x) - diag((bsxfun(quad_log_kernel, x, function_sample_points) * C) * bsxfun(quad_log_kernel, function_sample_points, x)'));

%exp_log_posterior = @(x)(exp(log_posterior(x))); % Construct posterior function.
exp_log_posterior = @(x)(exp(log_posterior(x) + 0.5.*log_posterior_variance(x))); % Construct posterior function, taking into account variance.

% Plot exp(likelihood-GP posterior).
exp_gpl_handle = plot( xrange, exp(log_posterior(xrange)), 'b-.'); hold on;
true_integral_of_log_gp = mean(exp(log_posterior(xrange)))
%pause;


% Without looking at the function again, model the difference between
% likelihood-GP posterior and exp(log-likelihood-GP posterior).
% =====================================================================

quad_diff_length_scale = .2;
quad_diff_kernel = @(x,y)exp( - 0.5 * (( x - y ) .^ 2 ) ./ quad_diff_length_scale );
quad_diff_noise = 1e-6;

% Choose surrogate function sample points.
n_diff_samples = 15;
diff_sample_points = linspace( -1, 6, n_diff_samples);
diff_values = exp_log_posterior(diff_sample_points') - posterior(diff_sample_points');
diff_points_handle = plot( diff_sample_points, diff_values, 'bd'); hold on;

K = bsxfun(quad_diff_kernel, diff_sample_points', diff_sample_points ); % Fill in gram matrix
C = inv( K + quad_diff_noise^2 .* diag(N) ); % Compute inverse covariance
weights = C * diff_values;  % Now compute kernel function weights.
diff_posterior = @(x)(bsxfun(quad_diff_kernel, diff_sample_points, x) * weights); % Construct posterior function.

% Plot difference.
diff_handle = plot( xrange, diff_posterior(xrange), 'b-'); hold on;
%pause;

% Final approximation is GP posterior plus exp(LGP posterior).
% ==============================================================
final = @(x)(posterior(x) + diff_posterior(x));
final_handle = plot( xrange, final(xrange), 'g-'); hold on;
better_gp_estimate = mean(final(xrange))

legend( [like_handle, sp_handle, gpf_handle, exp_gpl_handle, diff_points_handle, diff_handle, final_handle], ...
        { 'f(x)', 'function samples', 'gp on f(x) (bad model, but tractable)', 'exp(gp on log(f(x)) (good model, intractable)', 'difference samples (dont depend on f(x))', 'gp on diff (tractable)', 'final approx (tractable)' } );
