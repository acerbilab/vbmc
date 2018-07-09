% Bare-bones Demo of Bayesian Quadrature with Log-GPs,
% using the new linearization around the mean of the log-GP posterior.
%
% Takes a function that's better modeled in the log-domain and tries to
% integrate under it.
%
% David Duvenaud
% March 2012
% ===========================


% Define a likelihood function (one that is well-modeled by a gp on the log).
likelihood = @(x)(normpdf(x,4,.4).*40);
loglikelihood = @(x)log(likelihood(x));

% Plot likelihood.
N = 200;
xrange = linspace( 2.575, 5.5, N )';

clf;
col_width = 8.25381;  % ICML double column width in cm.
lw = 1.5;
fontsize = 12;
opacity = 0.1;

% Choose function sample points.
n_f_samples = 3;
function_sample_points = [ 3.1 3.6 4.7];

% Choose surrogate function sample points.
n_c_samples = 4;
c_sample_points = linspace( 3.2, 5.2, n_c_samples);

% Evaluate likelihood and log-likelood at sample points.
y = likelihood(function_sample_points)';
logy = loglikelihood(function_sample_points)';

%pause;
myeps = 0.05;

% Model likelihood with a GP.
% =================================

% Define quadrature hypers.
quad_length_scale = .3;
quad_kernel = @(x,y) 20*exp( - 0.5 * ( ( x - y ) .^ 2 ) ./ quad_length_scale );
quad_noise = 1e-6;

% Perform GP inference to get posterior mean function.
K = bsxfun(quad_kernel, function_sample_points', function_sample_points ); % Fill in gram matrix
C = inv( K + quad_noise^2 .* diag(ones(n_f_samples,1)) ); % Compute inverse covariance
weights = C * y;  % Now compute kernel function weights.
posterior = @(x)(bsxfun(quad_kernel, function_sample_points, x) * weights); % Construct posterior function.
posterior_variance = @(x)(bsxfun(quad_kernel, x, x) - diag((bsxfun(quad_kernel, x, function_sample_points) * C) * bsxfun(quad_kernel, function_sample_points, x)'));



% Fit a log-GP to the likelihood
% =================================
quad_log_length_scale = quad_length_scale * 4;
quad_log_kernel = @(x,y) 2.*exp( - 0.5 * (( x - y ) .^ 2 ) ./ quad_log_length_scale );
quad_log_noise = 1e-6;

K = bsxfun(quad_log_kernel, function_sample_points', function_sample_points ); % Fill in gram matrix
C = inv( K + quad_log_noise^2 .* diag(ones(n_f_samples,1)) ); % Compute inverse covariance
weights = C * logy;  % Now compute kernel function weights.
loggp_mean = @(x)(bsxfun(quad_log_kernel, function_sample_points, x) * weights); % Construct posterior function.
loggp_variance = @(x)(bsxfun(quad_log_kernel, x, x) - diag((bsxfun(quad_log_kernel, x, function_sample_points) * C) * bsxfun(quad_log_kernel, function_sample_points, x)'));



% Without looking at the function again, model the expectation of the
% log-GP posterior.
% =====================================================================

quad_v_length_scale = quad_length_scale / 2;
quad_v_kernel = @(x,y)20*exp( - 0.5 * (( x - y ) .^ 2 ) ./ quad_v_length_scale );
quad_v_noise = 1e-6;

% Evaluate at surrogate sample points and function sample points.
v_sample_points = [ c_sample_points function_sample_points] ;
n_v_samples = n_c_samples + n_f_samples;
v_values = exp(loggp_mean(v_sample_points') + 0.5.*loggp_variance(v_sample_points'));
v_log_values = loggp_mean(v_sample_points');

K = bsxfun(quad_v_kernel, v_sample_points', v_sample_points ); % Fill in gram matrix
C = inv( K + quad_v_noise^2 .* diag(ones(n_v_samples,1)) ); % Compute inverse covariance
weights = C * v_values;  % Now compute kernel function weights.
final = @(x)(bsxfun(quad_v_kernel, v_sample_points, x) * weights); % Construct posterior function.


% Overworld
% ==========================
ha1 = subaxis( 2, 1, 1,'SpacingVertical',0.1, 'MarginLeft', .1,'MarginRight',0);

like_handle = plot( xrange, likelihood(xrange), 'k', 'Linewidth', lw); hold on; % pause;
gpf_handle = plot( xrange, posterior(xrange), 'r', 'Linewidth', lw); hold on;
final_handle = plot( xrange, final(xrange), 'g-', 'Linewidth', lw); hold on;  % Plot likelihood-GP posterior.
diff_points_handle = plot( v_sample_points(1:n_c_samples), v_values(1:n_c_samples), 'ko','Markersize', 5); hold on;
sp_handle = plot( function_sample_points, y, 'kx', 'Linewidth', lw, 'MarkerSize', 10); hold on;

hl1 = legend( [ like_handle, sp_handle, gpf_handle, final_handle], ...
        {  '$\ell(x)$', '$\ell(x_s)$','$m(\ell(x))$', 'final approx' }, ...
        'Fontsize', fontsize, 'Location', 'EastOutside', 'Interpreter','latex');

legend boxoff  
set( gca, 'XTick', [] );
%set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
%set( gca, 'yTickLabel', '' );
xlabel( '$x$' );
ylabel( '$\ell(x)$\qquad' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
set(get(gca,'YLabel'),'Rotation',90,'Interpreter','latex', 'Fontsize', fontsize);
%set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off');
set(gcf, 'color', 'white');
set(gca, 'YGrid', 'off');
xlim([xrange(1) xrange(end)])




% Underworld
% ======================== 
ha2 = subaxis( 2, 1, 2,'SpacingVertical',0.1, 'MarginLeft', .1,'MarginRight',0);
log_like_handle = plot( xrange, log(likelihood(xrange)), 'k', 'Linewidth', lw); hold on; % pause;
gp_tl_handle = plot( xrange, loggp_mean(xrange), 'b-.', 'Linewidth', lw); hold on;
diff_points_handle = plot( v_sample_points(1:n_c_samples), v_log_values(1:n_c_samples), 'ko','Markersize', 5); hold on;
log_sp_handle = plot( function_sample_points, log(y), 'kx', 'Linewidth', lw, 'MarkerSize', 10); hold on;
hl2 = legend( [log_like_handle, log_sp_handle, gp_tl_handle, diff_points_handle], ...
        { '$\log\ell(x)$', '$\log\ell(x_s)$', '$m(\log \ell(x))$', '$\log\ell(x_c)$' }, ...
        'Fontsize', fontsize, 'Location', 'EastOutside','Interpreter','latex');
legend boxoff  

%line( [xrange(1), xrange(end)], [0 0], 'linestyle', '--', 'color', 'k', 'linewidth', lw );
    
set( gca, 'XTick', [] );
%set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
%set( gca, 'yTickLabel', '' );
xlabel( '$x$' );
ylabel( '$log\ell(x)$' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
set(get(gca,'YLabel'),'Rotation',90,'Interpreter','latex', 'Fontsize', fontsize);
%set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off');
set(gcf, 'color', 'white');
set(gca, 'YGrid', 'off');
xlim([xrange(1) xrange(end)])
ylim([-1 4.5])


pa1 = get(ha1,'Position')
pa2 = get(ha2,'Position')
set(ha2,'Position',[pa1(1) pa2(2) pa1(3) pa2(4)])

pl1 = get(hl1,'Position')
pl2 = get(hl2,'Position')
set(hl2,'Position',[pl1(1), pl1(2) - pa1(2) + pa2(2), pl2(3), pl2(4)])

set_fig_units_cm( 2.5 * col_width, 14 );
save2pdf('~/Dropbox/papers/Bayesian Quadrature/log_bmc_writeup/figures/loggp');


% Version with variances.
% ================================================================

figure(2);

% Overworld
% ==========================
ha1 = subaxis( 2, 1, 1,'SpacingVertical',0.1, 'MarginLeft', .1,'MarginRight',0);


gpf_handle = plot( xrange, posterior(xrange), 'r'); hold on;
jbfill( xrange', ...
        posterior(xrange)' + 2.*sqrt(posterior_variance(xrange))', ...
        posterior(xrange)' - 2.*sqrt(posterior_variance(xrange))', ...
        colorbrew(1), 'none', 1, opacity); hold on;

final_handle = plot( xrange, final(xrange), 'g-'); hold on;  % Plot likelihood-GP posterior.
jbfill( xrange', ...
        final(xrange)' + 2.*sqrt((final(xrange).^2).*loggp_variance(xrange))', ...
        final(xrange)' - 2.*sqrt((final(xrange).^2).*loggp_variance(xrange))', ...
        colorbrew(3), 'none', 1, opacity); hold on;
    
hl1 = legend( [ gpf_handle, final_handle], ...
        {  '$m(\ell(x))$', 'final approx' }, ...
        'Fontsize', fontsize, 'Location', 'EastOutside', 'Interpreter','latex');

legend boxoff  
set( gca, 'XTick', [] );
%set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
%set( gca, 'yTickLabel', '' );
xlabel( '$x$' );
ylabel( '$\ell(x)$\qquad' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
set(get(gca,'YLabel'),'Rotation',90,'Interpreter','latex', 'Fontsize', fontsize);
%set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off');
set(gcf, 'color', 'white');
set(gca, 'YGrid', 'off');
xlim([xrange(1) xrange(end)])


% Underworld
% ======================== 
ha2 = subaxis( 2, 1, 2,'SpacingVertical',0.1, 'MarginLeft', .1,'MarginRight',0);
%log_like_handle = plot( xrange, log(likelihood(xrange)), 'k', 'Linewidth', lw); hold on; % pause;
gp_tl_handle = plot( xrange, loggp_mean(xrange), 'b-', 'Linewidth', lw); hold on;
jbfill( xrange', ...
        loggp_mean(xrange)' + 2.*sqrt(loggp_variance(xrange))', ...
        loggp_mean(xrange)' - 2.*sqrt(loggp_variance(xrange))', ...
        colorbrew(2), 'none', 1, opacity); hold on;

diff_points_handle = plot( v_sample_points(1:n_c_samples), v_log_values(1:n_c_samples), 'ko','Markersize', 5); hold on;
log_sp_handle = plot( function_sample_points, log(y), 'kx', 'Linewidth', lw, 'MarkerSize', 10); hold on;
hl2 = legend( [ gp_tl_handle], ...
        {  '$m(\log \ell(x))$' }, ...
        'Fontsize', fontsize, 'Location', 'EastOutside','Interpreter','latex');
legend boxoff  

%line( [xrange(1), xrange(end)], [0 0], 'linestyle', '--', 'color', 'k', 'linewidth', lw );
    
set( gca, 'XTick', [] );
%set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
%set( gca, 'yTickLabel', '' );
xlabel( '$x$' );
ylabel( '$log\ell(x)$' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
set(get(gca,'YLabel'),'Rotation',90,'Interpreter','latex', 'Fontsize', fontsize);
%set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off');
set(gcf, 'color', 'white');
set(gca, 'YGrid', 'off');
xlim([xrange(1) xrange(end)])
ylim([-1 4.5])


pa1 = get(ha1,'Position')
pa2 = get(ha2,'Position')
set(ha2,'Position',[pa1(1) pa2(2) pa1(3) pa2(4)])

pl1 = get(hl1,'Position')
pl2 = get(hl2,'Position')
set(hl2,'Position',[pl1(1), pl1(2) - pa1(2) + pa2(2), pl2(3), pl2(4)])


set_fig_units_cm( 2.5 * col_width, 14 );
save2pdf('~/Dropbox/papers/Bayesian Quadrature/log_bmc_writeup/figures/loggp_var');

