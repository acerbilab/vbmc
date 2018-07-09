% A simple cartoon of Bayesian Monte Carlo.
%
% David Duvenaud
% February 2012
% ===========================


function bmc_intro


col_width = 8.25381;  % ICML double column width in cm.
lw = .5;  % line width
N = 2000;
xrange = linspace( 0, 25, N )';

% Choose function sample points.
function_sample_points = [ 5 12 16 ];
y = [ 2 8 4]';

noise_sds = [0.5 0.1 1]';

% Model function with a GP.
% =================================

% Define quadrature hypers.
quad_length_scale = 2;
quad_output_scale = 3;
quad_kernel = @(x,y)quad_output_scale * exp( -0.5 * ( ( x - y ) .^ 2 ) ./ exp(quad_length_scale) );
quad_kernel_dl = @(x,y)quad_output_scale*( -0.5 * ( ( x - y ) .^ 2 ) .* quad_kernel(x, y) ) ./ exp(quad_length_scale);
quad_kernel_at_data = @(x)(bsxfun(quad_kernel, x, function_sample_points));
quad_kernel_dl_at_data = @(x)(bsxfun(quad_kernel_dl, x, function_sample_points));

% Perform GP inference to get posterior mean function.
K = bsxfun(quad_kernel, function_sample_points', function_sample_points ); % Fill in gram matrix
C = inv( K + diag(noise_sds.^2) ); % Compute inverse covariance
weights = C * y;  % Now compute kernel function weights.
posterior = @(x)(bsxfun(quad_kernel, function_sample_points, x) * weights); % Construct posterior function.
posterior_variance = @(x)(bsxfun(quad_kernel, x, x) - diag((bsxfun(quad_kernel, x, function_sample_points) * C) * bsxfun(quad_kernel, function_sample_points, x)'));
K_dl = bsxfun(quad_kernel_dl, function_sample_points, function_sample_points');
dmu_dl = @(x)( ( quad_kernel_dl_at_data(x) * C - quad_kernel_at_data(x) * C * K_dl * C )) * y;  % delta in the mean function vs delta in lengthscale.


% Plot posterior variance.
clf;
edges = [posterior(xrange)+2*sqrt(posterior_variance(xrange)); flipdim(posterior(xrange)-2*sqrt(posterior_variance(xrange)),1)];
hc1 = fill([xrange; flipdim(xrange,1)], edges, [6.5 6.5 8]/8, 'EdgeColor', 'none'); hold on;

[h,g] = crosshatch_poly([xrange; flipdim(xrange,1)], [posterior(xrange); zeros(size(xrange))], -20, 1, ...
    'linestyle', '-', 'linecolor', [ .6 .6 .6 ], 'linewidth', lw, 'hold', 1);
fill( [xrange; flipdim(xrange,1)], [posterior(xrange); 10.*ones(size(xrange))], [ 1 1 1], 'EdgeColor', 'none');

edges = [posterior(xrange)+2*sqrt(posterior_variance(xrange)); flipdim(posterior(xrange),1)];
hc1 = fill([xrange; flipdim(xrange,1)], edges, [6.5 6.5 8]/8, 'EdgeColor', 'none'); hold on;


h1 = plot( xrange, posterior(xrange), 'b-', 'Linewidth', lw); hold on;
h2 = errorbar( function_sample_points, y, 2*noise_sds, 'kd', 'Marker', '.', ...
 'MarkerSize', 10, 'Linewidth', lw );
 %'Color', [0.6 0.6 0.6]..

% Add axes, legend, make the plot look nice, and save it.
xlim( [xrange(1) - 0.04, xrange(end)]);
ylim( [ -3 10] );
legend( [h2 h1 hc1 g(1)], ...
        {'samples', 'posterior mean', 'posterior variance', 'expected area'}, ...
        'Location', 'NorthEast', 'Fontsize', 6);
set( gca, 'XTick', [] );
set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
set( gca, 'yTickLabel', '' );
xlabel( '$x$' );
ylabel( '$\ell(x)$\qquad' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 8);
set(get(gca,'YLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', 8);
set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off');
set(gcf, 'color', 'white');
set(gca, 'YGrid', 'off');
legend boxoff

set_fig_units_cm( 10, 5.5 );
%matlabfrag('~/Dropbox/papers/sbq-paper/figures/bmc_intro');
%savepng('int_hypers');
%saveeps('int_hypers');
matlabfrag('/Volumes/UNTITLED/Documents/SBQ/bmc_intro');

end


