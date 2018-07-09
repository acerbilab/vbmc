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
set(0, 'defaulttextinterpreter', 'none')
t = @(x) log(x+1);

% Define a likelihood function (one that is well-modeled by a gp on the log).
likelihood = @(x)(normpdf(x,4,.4).*40);
loglikelihood = @(x)t(likelihood(x));

% Plot likelihood.
N = 2000;
xrange = linspace( 3.2, 4.9, N )'
%xrange = linspace( 3.075, 5.25, N )';

clf;
col_width = 15.5;  % NIPS column width in cm.
lw = 0.5;

% Choose function sample points.
n_f_samples = 3;
function_sample_points = [ 3.3 3.6 4.5];

% Evaluate likelihood and log-likelood at sample points.
y = likelihood(function_sample_points)';
logy = loglikelihood(function_sample_points)';

%pause;
myeps = 0.05;

% Model likelihood with a GP.
% =================================

% Define quadrature hypers.
quad_length_scale = 0.8;
quad_kernel = @(x,y)exp( - 0.5 * ( ( x - y ) .^ 2 ) ./ quad_length_scale );
quad_noise = 1e-6;

% Perform GP inference to get posterior mean function.
K = bsxfun(quad_kernel, function_sample_points', function_sample_points ); % Fill in gram matrix
C = inv( K + quad_noise^2 .* diag(ones(n_f_samples,1)) ); % Compute inverse covariance
weights = C * y;  % Now compute kernel function weights.
posterior = @(x)(bsxfun(quad_kernel, function_sample_points, x) * weights); % Construct posterior function.



% Model log-likelihood with a GP
% =================================
quad_log_length_scale = 1.2;
quad_log_kernel = @(x,y)exp( - 0.5 * (( x - y ) .^ 2 ) ./ quad_log_length_scale );
quad_log_noise = 1e-6;

K = bsxfun(quad_log_kernel, function_sample_points', function_sample_points ); % Fill in gram matrix
C = inv( K + quad_log_noise^2 .* diag(ones(n_f_samples,1)) ); % Compute inverse covariance
weights = C * logy;  % Now compute kernel function weights.
log_posterior = @(x)(bsxfun(quad_log_kernel, function_sample_points, x) * weights); % Construct posterior function.
log_posterior_variance = @(x)(bsxfun(quad_log_kernel, x, x) - diag((bsxfun(quad_log_kernel, x, function_sample_points) * C) * bsxfun(quad_log_kernel, function_sample_points, x)'));



% Without looking at the function again, model the difference between
% likelihood-GP posterior and exp(log-likelihood-GP posterior).
% =====================================================================

quad_diff_length_scale = quad_log_length_scale / 4;
quad_diff_kernel = @(x,y)exp( - 0.5 * (( x - y ) .^ 2 ) ./ quad_diff_length_scale );
quad_diff_noise = 1e-6;

% Choose surrogate function sample points.
n_diff_samples = 6;
diff_sample_points = [linspace( 3.2, 5.2, n_diff_samples) function_sample_points] ;
n_diff_samples = n_diff_samples + n_f_samples;
diff_values = log_posterior(diff_sample_points') - t(max(myeps,posterior(diff_sample_points')));

K = bsxfun(quad_diff_kernel, diff_sample_points', diff_sample_points ); % Fill in gram matrix
C = inv( K + quad_diff_noise^2 .* diag(ones(n_diff_samples,1)) ); % Compute inverse covariance
weights = C * diff_values;  % Now compute kernel function weights.
delta = @(x)(bsxfun(quad_diff_kernel, diff_sample_points, x) * weights); % Construct posterior function.


% Final approximation is GP posterior plus exp(LGP posterior).
% ==============================================================
%final = @(x)(posterior(x) + diff_posterior(x));
final = @(x)(posterior(x).*(1 + delta(x)));


% Overworld
% ==========================
subaxis( 1, 2, 1,'SpacingVertical',0.1, 'MarginLeft', 0.1,'MarginRight',0);

like_handle = plot( xrange, likelihood(xrange), 'k'); hold on; % pause;
sp_handle = plot( function_sample_points, y, 'k.', 'Markersize', 10); hold on;
% Plot likelihood-GP posterior.
gpf_handle = plot( xrange, posterior(xrange), 'r'); hold on;

%exp_gpl_handle = plot( xrange, exp(log_posterior(xrange)), 'b-.'); hold on;

final_handle = plot( xrange, final(xrange), 'g-'); hold on;

legend( [ sp_handle, like_handle, gpf_handle, final_handle], ...
        { '$\ell(x_s)$', '$\ell(x)$', '$m\bigl(\ell(x)\bigr)$', 'final approx' }, ...
        'Fontsize', 8, 'Location', 'EastOutside');
legend boxoff  

set( gca, 'XTick', [] );
%set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
%set( gca, 'yTickLabel', '' );
xlabel( '$x$' );
ylabel( '$\ell(x)$\qquad' );
set(get(gca,'XLabel'),'Rotation',0, 'Fontsize', 8);
set(get(gca,'YLabel'),'Rotation',90, 'Fontsize', 8);
%set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off');
set(gcf, 'color', 'white');
set(gca, 'YGrid', 'off');
xlim([xrange(1) xrange(end)])




% Underworld
% ======================== 
subaxis( 1, 2, 2,'SpacingVertical',0.1, 'MarginLeft',-0.05,'MarginRight',0);

% Plot exp(likelihood-GP posterior).

log_like_handle = plot( xrange, t(likelihood(xrange)), 'k'); hold on; % pause;
log_sp_handle = plot( function_sample_points, t(y), 'k.', 'Markersize', 10); hold on;
log_gpf_handle = plot( xrange, t(max(myeps,posterior(xrange))), 'r'); hold on;
gp_tl_handle = plot( xrange, log_posterior(xrange), 'b-.'); hold on;
%diff_points_handle = plot( diff_sample_points, diff_values', 'ko','Markersize', 5); hold on;
%delta_handle = plot( xrange, delta(xrange), 'b-'); hold on;

legend( [log_sp_handle, log_like_handle, log_gpf_handle, gp_tl_handle, diff_points_handle, delta_handle], ...
        { '$\log \ell(x_s)$', ...
        '$\log \ell(x)$', ...
        '$\log m\bigl(\ell(x)|\ell(x_s)\bigr)$', ...
        '$m\bigl(\log \ell(x)|\log \ell(x_s)\bigr)$',...
        '$\Delta_c$', ...
        '$m\bigl(\Delta_{\ll|s}(x)|\Delta_c\bigr)$' } ...
        , 'Fontsize', 8, 'Location', 'EastOutside');
legend boxoff  

line( [xrange(1), xrange(end)], [0 0], 'linestyle', '--', 'color', 'k', 'linewidth', lw );
    
set( gca, 'XTick', [] );
%set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
%set( gca, 'yTickLabel', '' );
xlabel( '$x$' );
ylabel( '$\log \ell(x)$' );
set(get(gca,'XLabel'),'Rotation',0, 'Fontsize', 8);
set(get(gca,'YLabel'),'Rotation',90, 'Fontsize', 8);
%set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off');
set(gcf, 'color', 'white');
set(gca, 'YGrid', 'off');
xlim([xrange(1) xrange(end)])
ylim([0.5 4.5])


%position = get(gca, 'Position');          % Width of the figure
%tightinset = get(gca, 'TightInset');      % Width of the surrounding text
%total_width = position(3) + tightinset(3) + tightinset(1);
%scale_factor = 1;%(pagewidth*fraction)/total_width;
%set(gca, 'Position', [position(1:2), position(3:4).*scale_factor]);

set_fig_units_cm( 8, 6 );
%matlabfrag('~/Dropbox/papers/sbq-paper/figures/delta');  

set_fig_units_cm( col_width-.5, 3 );
set(0, 'defaulttextinterpreter', 'none')
matlabfrag('~/Docs/sbq-paper/figures/deltav2');  

