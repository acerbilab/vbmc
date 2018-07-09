% Bare-bones Demo of computing marginal functions
% using Bayesian Quadrature with untransformed GPs.
%
% David Duvenaud
% March 2012
% ===========================

% Set random seed.
seed=0;
randn('state',seed);
rand('state',seed);

% Plotting options.
col_width = 8.25381;  % ICML double column width in cm.
lw = 1.5;
fontsize = 14;
opacity = 0.1;

% Define a 2D function.
D = 2;
mu1 = [-0.3 -0.3];
sigma1 = [.1 0; 0 .1];
mu2 = [0.2 0.8];
sigma2 = [.42 0; 0 .12];
f = @(x)(mvnpdf(x,mu1,sigma1) + mvnpdf(x,mu2,sigma2));

% generate a grid
n_xrange = 100;
xrange = linspace(-2,2,n_xrange);
[a,b] = meshgrid(xrange, xrange);
xstar = [a(:) b(:)];
n_xstar = size(xstar, 1);
fstar_vals = f(xstar);

% Plot the function.
%figure(1); clf;
%h1 = surf(a,b,reshape(fstar_vals, length( xrange), length( xrange) ), ...
%    'EdgeColor','none','LineStyle','none','FaceLighting','phong');  hold on;
%title('True function');


% Choose some points to sample.
prior_mu = [0 0];
prior_sigma = [ .3 0; 0 .3 ];
N = 4;
samples = mvnrnd(prior_mu, prior_sigma, N);
%samples = [ones(N,1), (1:N)' ./2 - 3];
sample_f_vals = f(samples);  %ones(N,1);  % Evaluate function at sample points.
%sp_handle = plot3( samples(:, 1), samples(:, 2), sample_f_vals, 'gx' );

% Model likelihood with a GP.
% =================================

% Define quadrature hypers.
length_scales = [ 0.5; 0.6 ];
quad_sigma = diag(ones(D, 1) .* length_scales);
quad_height = 2;
quad_kernel = @(x,y) quad_height*mvnpdf(x, y, quad_sigma);
quad_noise = 1e-6;
quad_sigma_x = quad_sigma(1,1);
quad_sigma_y = quad_sigma(2,2);

% Perform GP inference to get posterior mean function.
K = NaN(N, N);
for i = 1:N
    for j = 1:N
        K(i,j) = quad_kernel( samples(i,:), samples(j, :));
    end
end
%K = bsxfun(quad_kernel, permute(xstar, [1,3,2]), permute(xstar, [3,1,2]) ); % Fill in gram matrix
%K = reshape(K, n_xstar, n_xstar );
Kinv = inv( K + quad_noise^2 .* diag(ones(length(N),1)) ); % Compute inverse covariance
weights = Kinv * sample_f_vals;  % Now compute kernel function weights.
posterior = @(x)(quad_kernel(samples, x)' * weights); % Construct posterior function.
posterior_variance = @(x)quad_kernel(x, x) - diag(quad_kernel(x, samples)' * Kinv * quad_kernel( samples, x));


% Plot the posterior mean and variance.
mu_vals = NaN(n_xrange, n_xrange);
var_vals = NaN(n_xrange, n_xrange);
for i = 1:n_xrange
    for j = 1:n_xrange
        mu_vals(i,j) = posterior( [xrange(j), xrange(i)]);
        var_vals(i,j) = posterior_variance( [xrange(j), xrange(i)]);
    end
end

figure(2); clf;
transparency = 0.3;
h_2d_mean = surf(a,b,mu_vals, ...
    'EdgeColor','none','LineStyle','none','FaceLighting','phong',  'FaceAlpha', 0.8);  hold on;

% Plot 2d marginal variance.
%h_2d_var = surf(a,b,mu_vals + 2.*sqrt(var_vals), ...
%    'EdgeColor','none','LineStyle','none','FaceAlpha',transparency,'EdgeAlpha',transparency);  hold on;
%h3 = surf(a,b,mu_vals - 2.*sqrt(var_vals), ...
%    'EdgeColor','none','LineStyle','none','FaceAlpha',transparency,'EdgeAlpha',transparency);  hold on;

% Plot data locations.
sp_handle = plot3( samples(:, 1), samples(:, 2), sample_f_vals, ...
    'kx', 'LineWidth', lw ); hold on;
set(sp_handle, 'MarkerSize', 6);

%  Show prior isocontours
n_contours = 3;
contour_radiuses = [3 2 1];
points_per_contour = 300;
angles = linspace(0, 2*pi, points_per_contour);
for c_ix = 1:n_contours
    ix = 1;
    for angle = angles
        cur_pos =  [cos(angle) sin(angle)];
        mh_dst = sqrt( cur_pos * inv(prior_sigma) * cur_pos');
        contour_points(ix, 1) = prior_mu(1) + cos(angle) / mh_dst * contour_radiuses(c_ix);
        contour_points(ix, 2) = prior_mu(2) + sin(angle) / mh_dst * contour_radiuses(c_ix);
        ix = ix + 1;
    end
    prior_h = plot3( contour_points(:, 1), contour_points(:, 2), -3.*ones(ix -1, 1), 'Color', colorbrew(1).^(2*c_ix) ); hold on;
end


% Compute posterior marginals.
% =================================

marginal_posterior_x = @(xstar) bmc_marginal_mean( prior_mu(2), prior_sigma(2,2), Kinv, samples(:, 1), samples(:, 2), sample_f_vals, ...
                                               quad_sigma_x, quad_sigma_y, quad_height, xstar );
marginal_posterior_y = @(xstar) bmc_marginal_mean( prior_mu(1), prior_sigma(1,1), Kinv, samples(:, 2), samples(:, 1), sample_f_vals, ...
                                               quad_sigma_y, quad_sigma_x, quad_height, xstar );                                       
posterior_variance_x = @(xstar) bmc_marginal_variance( prior_mu(2), prior_sigma(2,2), Kinv, samples(:, 1), samples(:, 2), ...
                                               quad_sigma_x, quad_sigma_y, quad_height, xstar );
posterior_variance_y = @(xstar) bmc_marginal_variance( prior_mu(1), prior_sigma(1,1), Kinv, samples(:, 2), samples(:, 1), ...
                                               quad_sigma_y, quad_sigma_x, quad_height, xstar );
                                          
% Plot marginals
% ==========================
marg_location = xrange(end) + 0.5;

marginal_x_vals = marginal_posterior_x(xrange');
marginal_x_variance = posterior_variance_x(xrange');

marginal_y_vals = marginal_posterior_y(xrange');
marginal_y_variance = posterior_variance_y(xrange');

marg_handle_x = plot3( xrange, marg_location.*ones(size(xrange)), marginal_x_vals, '-', ...
                       'Color', colorbrew(2), 'Linewidth', lw); hold on;
marg_varhandle_x = fill3([xrange'; xrange(end:-1:1)'], ...
        marg_location.*ones(1, 2*size(xrange,2)), ...
        [marginal_x_vals' + 2 .* marginal_x_variance', ...
         marginal_x_vals(end:-1:1)' - 2 .* marginal_x_variance(end:-1:1)'], ...
        colorbrew(2), 'EdgeColor', 'none', 'FaceAlpha', opacity * 3);

marg_handle_y = plot3( marg_location.*ones(size(xrange)), xrange, marginal_y_vals, '-', ...
                       'Color', colorbrew(3), 'Linewidth', lw); hold on;
marg_varhandle_y = fill3(marg_location.*ones(1, 2*size(xrange,2)), ...
        [xrange'; xrange(end:-1:1)'], ...
        [marginal_y_vals' + 2 .* marginal_y_variance', ...
         marginal_y_vals(end:-1:1)' - 2 .* marginal_y_variance(end:-1:1)'], ...
        colorbrew(3), 'EdgeColor', 'none', 'FaceAlpha', opacity * 3);

% Plot the data on top of each marginal.
% Todo: size according to its prob. under the prior.
marg_data_x = plot3( samples(:, 1), marg_location*ones(1, N), sample_f_vals, ...
    'x', 'Color', colorbrew(2).^2); hold on;
marg_data_y = plot3( marg_location*ones(1, N), samples(:, 2), sample_f_vals, ...
    'x', 'Color', colorbrew(3).^2);
    
hl1 = legend( [ sp_handle, h_2d_mean, prior_h, marg_handle_x, marg_varhandle_x, marg_handle_y, marg_varhandle_y ], ...
        { 'sample points', '2d posterior mean', ...
        'prior on (x,y)',  'marginal mean of x','marginal variance of x'...
        'marginal mean of y','marginal variance of y'}, ...
        'Fontsize', 10, 'Location', 'EastOutside' );
legend boxoff

set( gca, 'xTick', [] );
set( gca, 'yTick', [] );
set( gca, 'zTick', [] );
set( gca, 'xTickLabel', '' );
set( gca, 'yTickLabel', '' );
set( gca, 'zTickLabel', '' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
set(get(gca,'YLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
set(get(gca,'ZLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);

set(gca, 'Box', 'off');
set(gcf, 'color', 'white');
set(gca, 'YGrid', 'off');
zlim([-3 2]);
xlabel('x');
ylabel('y');
zlabel('z');
xlim([xrange(1) marg_location]);
ylim([xrange(1) marg_location]);
view( [-47 46])

set_fig_units_cm(30, 20 );
save2pdf('~/Dropbox/papers/bayesian_quadrature/marginals/figures/2d_marginals.pdf', gcf, 300, true);
