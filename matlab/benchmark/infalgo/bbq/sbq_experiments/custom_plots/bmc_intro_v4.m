% A simple cartoon of Bayesian Monte Carlo.
%
% This version also shows the posterior over evidence.
%
% This version is for the herding paper.
%
% David Duvenaud
% February 2012
% ===========================


function bmc_intro_v4

close all;

fontsize = 10;

% Plot our function.
N = 2000;
xrange = linspace( 0, 25, N )';

% Choose function sample points.
function_sample_points = [ 5 13 16 ];
y = [ 6 8 4]';

prior.mean = 10;
prior.covariance = 8;
prior.plot_scale = 70;

% Model function with a GP.
% =================================

% Define quadrature hypers.
quad_length_scale = 2;
quad_kernel = @(x,y)exp( -0.5 * ( ( x - y ) .^ 2 ) ./ exp(quad_length_scale) );
quad_noise = 1e-6;

% Perform GP inference to get posterior mean function.
K = bsxfun(quad_kernel, function_sample_points', function_sample_points ); % Fill in gram matrix
C = inv( K + quad_noise^2 .* diag(N) ); % Compute inverse covariance
weights = C * y;  % Now compute kernel function weights.
posterior = @(x)(bsxfun(quad_kernel, function_sample_points, x) * weights); % Construct posterior function.
posterior_variance = @(x)(bsxfun(quad_kernel, x, x) - diag((bsxfun(quad_kernel, x, function_sample_points) * C) * bsxfun(quad_kernel, function_sample_points, x)'));

color_ix = 3;
% Plot posterior variance.
clf;
%subplot(2, 2, 1);

fillcolor = sqrt(colorbrew(3));%[6 6 8]/8;
edges = [posterior(xrange)+2*sqrt(posterior_variance(xrange)); flipdim(posterior(xrange)-2*sqrt(posterior_variance(xrange)),1)];
h_fill1 = fill([xrange; flipdim(xrange,1)], edges, sqrt(fillcolor), 'EdgeColor', 'none'); hold on;

[h,g] = crosshatch_poly([xrange; flipdim(xrange,1)], [posterior(xrange); zeros(size(xrange))], -45, 1, ...
    'linestyle', '-', 'linecolor', [0.5 0.5 0.5], 'linewidth', 1, 'hold', 1);
fill( [xrange; flipdim(xrange,1)], [posterior(xrange); 10.*ones(size(xrange))], [ 1 1 1], 'EdgeColor', 'none');

edges = [posterior(xrange)+2*sqrt(posterior_variance(xrange)); flipdim(posterior(xrange),1)];
h_fill2 = fill([xrange; flipdim(xrange,1)], edges, sqrt(fillcolor), 'EdgeColor', 'none'); hold on;


h_postmean = plot( xrange, posterior(xrange), '-', 'Linewidth', 1.5, ...
                   'Color', fillcolor.^2); hold on;
h_points = plot( function_sample_points, y, 'kx', ...
 'MarkerSize', 7.5, 'Linewidth', 1 );
 %'Color', [0.6 0.6 0.6]..

 % Plot input distribution.
h_prior = plot( xrange, ...
                prior.plot_scale .* mvnpdf(xrange, prior.mean, prior.covariance), ...
                '--', 'Linewidth', 1, 'Color', colorbrew(2)); hold on;
 

% Do BMC
covfunc = @covSEiso;
init_hypers.lik = log(quad_noise);
init_hypers.cov = log([quad_length_scale 1 ]);
[expected_Z, Z_variance] = ...
    bmc_integrate(function_sample_points', y, prior, covfunc, init_hypers, false);


zplot_width = 4;

yrange = [ -3 10];
ylim( yrange );
xlim( [xrange(1) - 0.1, xrange(end)]);
%subplot(1, 2, 2);

zplot_left = xrange(end) + 0.1;

yvals = linspace(0, yrange(2), 600);
yplot_scale = 0.95*zplot_width / mvnpdf(0, 0, Z_variance);
post_h = plot( -mvnpdf(yvals', expected_Z, Z_variance)...
              .*yplot_scale + zplot_left, yvals, 'Color', colorbrew(1), 'Linewidth', .5);
line( [zplot_left zplot_left], [0 yrange(2)], 'Color', 'k')


% Add axes, legend, make the plot look nice, and save it.
%xlim( [xrange(1) - 0.04, xrange(end)]);

hAx = gca;
hLgnd = legend( [h_points h_postmean h_fill1 ], ...
    {'samples', 'GP mean', 'GP variance' },...
    'Location', 'SouthOutside', 'Fontsize', fontsize, 'Interpreter','latex');
legend boxoff
%set( h_l, 'Position', axes_pos );

set(hLgnd, 'Units','pixels')
op = get(hLgnd,'OuterPosition');
set(hLgnd, 'Units','normalized')

%# resize the plot axis vertically to make room for the legend
set(hAx, 'Units','pixels')
pos = get(hAx,'Position');
ins = get(hAx,'TightInset');
set(hAx, 'Position',[pos(1) pos(2)+op(4)- 50 pos(3) pos(4)-op(4) + 50])
set(hAx, 'Units','normalized')

%# move the legend to the bottom in the free space
set(hLgnd, 'Units','pixels');
legendpos = [pos(1) + 20, (pos(2)-ins(2))/2 - 30, op(3), op(4)];
set(hLgnd, 'OuterPosition',legendpos);
set(hLgnd, 'Units','normalized');


%lh=findall(gcf,'tag','legend');
%     lp=get(lh,'position');
%     set(lh,'position',[.1,-1,lp(3:4)]);
set( gca, 'XTick', [] );
set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
set( gca, 'yTickLabel', '' );
%set(gca,'xColor','r');
%set(gca,'yColor','k');
%set(gca,'XAxisLocation',0)
%a=axes('xcolor',get(f,'color'),'xtick',[])
xlabel( '$x$' );
ylabel( '$f(x)$\qquad' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
set(get(gca,'YLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off');
set(gcf, 'color', 'white');
set(gca, 'YGrid', 'off');


% Make a second set of axes for the right-hand side label.
ah=axes('position',get(gca,'position'),...
        'YAxisLocation','right',...
        'Color','none',...
        'visible','on');
set( gca, 'XTick', [] );
set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
set( gca, 'yTickLabel', '' );
%set(gca,'xColor','g');
%set(gca,'yColor','k');
set( get(gca,'YLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize', fontsize);
ylabel( '\quad$Z$' );
          
legendpos(1) = legendpos(1) + legendpos(3) * 2;           
hLgnd2 = legend( ah, [h_prior,g(1) , post_h], ...
    {'target density', 'expected area', 'posterior on Z \quad'}, ...
    'Location', legendpos, 'Fontsize', fontsize, 'Interpreter','latex');
legend boxoff
set(hLgnd2, 'Units','pixels');
set(hLgnd2, 'OuterPosition',legendpos);
set(hLgnd2, 'Units','normalized');

set_fig_units_cm( 6, 6 );
matlabfrag('/Volumes/UNTITLED/Documents/SBQ/');
%save2pdf('~/Dropbox/papers/herding-bmc/figures/bq_intro4.pdf', gcf, 300, true);
end


