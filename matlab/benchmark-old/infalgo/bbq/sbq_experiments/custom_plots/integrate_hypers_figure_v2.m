% Bare-bones Demo of Bayesian Quadrature when integrating out hypers,
% assuming that the posterior mean is linear in the hypers, and that
% the posterior distribution over hypers is normal, centered at the point used
% to calculate the postrior mean.
%
% Written to help understand Mike Osborne's AISTATS paper.
%
% David Duvenaud
% Jan 2012
% ===========================


function integrate_hypers_figure

randn('state', 0);
rand('twister', 0);  
    
clf;
col_width = 8.25381;  % ICML double column width in cm.
lw = 0.5;

% Plot our function.
N = 200;
xrange = linspace( 0, 25, N )';

% Choose function sample points.
function_sample_points = [ 12 16 ];
y = [ 8 4]';


% Model function with a GP.
% =================================

% Define quadrature hypers.
quad_length_scale = 1.25;
quad_kernel = @(x,y)exp( -0.5 * ( ( x - y ) .^ 2 ) ./ exp(quad_length_scale) );
quad_kernel_dl = @(x,y)( -0.5 * ( ( x - y ) .^ 2 ) .* quad_kernel(x, y) ) ./ exp(quad_length_scale);
quad_kernel_at_data = @(x)(bsxfun(quad_kernel, x, function_sample_points));
quad_kernel_dl_at_data = @(x)(bsxfun(quad_kernel_dl, x, function_sample_points));
quad_noise = 1e-6;

% Perform GP inference to get posterior mean function.
K = bsxfun(quad_kernel, function_sample_points', function_sample_points ); % Fill in gram matrix
C = inv( K + quad_noise^2 .* diag(N) ); % Compute inverse covariance
weights = C * y;  % Now compute kernel function weights.
posterior = @(x)(bsxfun(quad_kernel, function_sample_points, x) * weights); % Construct posterior function.
posterior_variance = @(x)(bsxfun(quad_kernel, x, x) - diag((bsxfun(quad_kernel, x, function_sample_points) * C) * bsxfun(quad_kernel, function_sample_points, x)'));
K_dl = bsxfun(quad_kernel_dl, function_sample_points, function_sample_points');
dmu_dl = @(x)( ( quad_kernel_dl_at_data(x) * C - quad_kernel_at_data(x) * C * K_dl * C )) * y;  % delta in the mean function vs delta in lengthscale.

% Check derivative
%quad_kernel2 = @(x,y)exp( -0.5 * ( ( x - y ) .^ 2 ) ./ exp(quad_length_scale + 0.00001) );
%K2 = bsxfun(quad_kernel2, function_sample_points', function_sample_points ); % Fill in gram matrix
%C2 = inv( K2 + quad_noise^2 .* diag(N) ); % Compute inverse covariance
%weights2 = C2 * y;  % Now compute kernel function weights.
%posterior2 = @(x)(bsxfun(quad_kernel2, function_sample_points, x) * weights2); % Construct posterior function.
%t = (posterior(xrange) - posterior2(xrange)) ./ 0.00001;

% Evaluate gradient of mean w.r.t. lengthscale at each point.
c_theta0 = dmu_dl(xrange);
varscale = 1;  % proportional to posterior variance in lengthscale;
extra_var = c_theta0.^2 .* varscale;



% Actually integrate lengthscale.
theta_mu = quad_length_scale;
theta_sigma = sqrt(varscale);

num_outer_samples = 100;
num_inner_samples = 10;
cdf_vals = linspace(0,1, num_outer_samples + 2);
cdf_vals = cdf_vals(2:end-1);  % don't include 0 and 1

cur_sample = 1;
f_history = NaN(num_outer_samples * num_inner_samples, length(xrange));
f_mu_history = NaN(num_outer_samples * num_inner_samples, length(xrange));

for i = 1:length(cdf_vals)

    % sample a lengthscale
    cur_lengthscale = norminv( cdf_vals(i), theta_mu, theta_sigma );
    
    % Recalculate process posterior.
    quad_kernel = @(x,y)exp( -0.5 * ( ( x - y ) .^ 2 ) ./ exp(cur_lengthscale) );
    K = bsxfun(quad_kernel, function_sample_points', function_sample_points ); % Fill in gram matrix
    Cinv = K + quad_noise^2 .* diag(N);
    weights = Cinv \ y;  % Now compute kernel function weights.

    posterior = @(x)(bsxfun(quad_kernel, function_sample_points, x) * weights); % Construct posterior function.
    posterior_covariance = @(x)(bsxfun(quad_kernel, x', x) - (bsxfun(quad_kernel, x, function_sample_points) / Cinv) * bsxfun(quad_kernel, function_sample_points, x)');
    
    % Evaluate posterior mean and covariance.
    fmu = posterior(xrange);
    fsigma = posterior_covariance(xrange);
      
    fprintf('.');  % Progress.
    
    for j = 1:num_inner_samples
        f_history(cur_sample, :) = mvnrnd(fmu, fsigma);
        f_mu_history(cur_sample, : ) = fmu;
        cur_sample = cur_sample + 1;
    end
end


% Plot posterior variance.
clf;
edges = [posterior(xrange)+2*sqrt(posterior_variance(xrange)); flipdim(posterior(xrange)-2*sqrt(posterior_variance(xrange)),1)];
edges2 = [posterior(xrange)+2*sqrt(posterior_variance(xrange) + extra_var); flipdim(posterior(xrange)-2*sqrt(posterior_variance(xrange) + extra_var),1)];
hc2 = fill([xrange; flipdim(xrange,1)], edges2, [6 8 6]/8, 'EdgeColor', 'none'); hold on;
hc1 = fill([xrange; flipdim(xrange,1)], edges, [6 6 8]/8, 'EdgeColor', 'none'); hold on;
h1 = plot( xrange, posterior(xrange), 'b-', 'Linewidth', lw); hold on;
h2 = plot( function_sample_points, y, 'kd', 'Marker', '.', ...
 'MarkerSize', 5, 'Linewidth', lw );
emp = plot(xrange, posterior(xrange) + 2.*std(f_mu_history)', 'k', 'Linewidth', lw ); hold on;
emp = plot(xrange, posterior(xrange) - 2.*std(f_mu_history)', 'k', 'Linewidth', lw ); hold on;

% Add axes, legend, make the plot look nice, and save it.
xlim( [xrange(1) - 0.04, xrange(end)]);
ylim( [ -8 14] );
legend_handle = legend( [h2 h1 hc1 hc2 emp], {'data', 'mean', 'variance', 'approx. marginalised length scale', 'true marginalised length scale'}, 'Location', 'SouthWest', 'Fontsize', 6);
set( gca, 'XTick', [] );
set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
set( gca, 'yTickLabel', '' );
xlabel( '$x$' );
ylabel( '$f(x)$\qquad' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 8);
set(get(gca,'YLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 8);
set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off');
set(gcf, 'color', 'white');
set(gca, 'YGrid', 'off');
legend boxoff

set_fig_units_cm( col_width, 5 );
matlabfrag('~/Dropbox/papers/sbq-paper/figures/int_hypers2');
%savepng('int_hypers');
%saveeps('int_hypers');

end


