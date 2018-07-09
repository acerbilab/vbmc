% A simple cartoon of Bayesian Monte Carlo.
%
% This version also shows the posterior over evidence.
%
% This version is for the herding paper.
%
% David Duvenaud February 2012 ===========================


function bmc_intro_v6

close all;

fontsize = 12;

% Plot our function.
N = 200;
xrange = linspace( 0, 25, N )';

% Choose function sample points.
function_sample_points = [ 5 13 16 ];
y = [ 6 8 4]';

prior.mean = 10;
prior.covariance = 8;
prior.plot_scale = 70;

% Model function with a GP. =================================

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

fillcolor = sqrt(colorbrew(color_ix));%[6 6 8]/8;
edges = [posterior(xrange)+2*sqrt(posterior_variance(xrange)); flipdim(posterior(xrange)-2*sqrt(posterior_variance(xrange)),1)];
h_fill1 = fill([xrange; flipdim(xrange,1)], edges, sqrt(fillcolor), 'EdgeColor', 'none'); hold on;

[h,g] = crosshatch_poly([xrange; flipdim(xrange,1)], [posterior(xrange); zeros(size(xrange))], 0, 1, ...
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
% h_prior = plot( xrange, ...
%                 prior.plot_scale .* mvnpdf(xrange, prior.mean, prior.covariance), ...
%                 '--', 'Linewidth', 1, 'Color', colorbrew(2)); hold on;
 

% Do BMC
covfunc = @covSEiso;
init_hypers.lik = log(quad_noise);
init_hypers.cov = log([quad_length_scale 1 ]);
[expected_Z, Z_variance] = ...
    bmc_integrate(function_sample_points', y, prior, covfunc, init_hypers, false);

x = function_sample_points;
Kfn = @(x, xd) bsxfun(quad_kernel, x', xd );
x_st = linspace(0,25,1000);
C_st = Kfn(x_st, x_st ) - Kfn(x_st, x) * C * Kfn(x, x_st);
m_st = Kfn(x_st, x) * weights;

num_draws = 3;
Z_draw = nan(num_draws, 1);
color_brew_inds = setdiff(1:num_draws+1, color_ix);
draw_h = nan(3,1);  
for i = 1:num_draws
    y_st = mvnrnd(m_st, C_st);
    draw_h(i) = plot(x_st, y_st, '-', 'Color', colorbrew(color_brew_inds(i)), 'Linewidth', 0.5);
    Z_draw(i) = bmc_integrate(x_st', y_st', prior, covfunc, init_hypers, false);
end



zplot_width = 4;

yrange = [ -3 10];
ylim( yrange );
xlim( [xrange(1) - 0.1, xrange(end)]);
%subplot(1, 2, 2);

zplot_left = xrange(end) * 0.9;

lower = 3; upper = yrange(2) - 1;
yvals = linspace(lower, upper, 600);
yplot_scale = 0.95*zplot_width / mvnpdf(0, 0, Z_variance);

post_h = plot( -mvnpdf(yvals', expected_Z + 1.5, Z_variance)...
              .*yplot_scale + zplot_left, yvals, '--', ...
              'Color', fillcolor, 'Linewidth', 1.5);

        
for i = 1:num_draws
plot( -mvnpdf(Z_draw(i), expected_Z + 1.5, Z_variance)...
  .*yplot_scale + zplot_left, Z_draw(i), '.', ...
  'Color',colorbrew(color_brew_inds(i)) , 'MarkerSize', 15);
end

plot( -mvnpdf(expected_Z+1.5, expected_Z + 1.5, Z_variance)...
  .*yplot_scale + zplot_left, expected_Z+1.5, '.', ...
  'Color',[0.5 0.5 0.5] , 'MarkerSize', 15);
          
line( [zplot_left, zplot_left], [lower, upper], 'Color', 'k');
tz = text( zplot_left + 0.9, expected_Z + 1.5, 'Z' );
set(tz, 'Interpreter', 'Latex' );




% Add axes, legend, make the plot look nice, and save it.
%xlim( [xrange(1) - 0.04, xrange(end) + 1]);

%hAx = gca; set( h_l, 'Position', axes_pos );

%set(hLgnd, 'Units','pixels') op = get(hLgnd,'OuterPosition'); set(hLgnd,
%'Units','normalized')

%# resize the plot axis vertically to make room for the legend
%set(hAx, 'Units','pixels') pos = get(hAx,'Position'); ins =
%get(hAx,'TightInset'); set(hAx, 'Position',[pos(1) pos(2)+op(4)- 50 pos(3)
%pos(4)-op(4) + 50]) set(hAx, 'Units','normalized')

%# move the legend to the bottom in the free space
%set(hLgnd, 'Units','pixels'); legendpos = [pos(1), pos(2), op(3), op(4)];
%set(hLgnd, 'OuterPosition',legendpos); set(hLgnd, 'Units','normalized');




%lh=findall(gcf,'tag','legend');
%     lp=get(lh,'position'); set(lh,'position',[.1,-1,lp(3:4)]);
set( gca, 'XTick', [] );
set( gca, 'yTick', [] );
set( gca, 'XTickLabel', '' );
set( gca, 'yTickLabel', '' );
%set(gca,'xColor','r'); set(gca,'yColor','k'); set(gca,'XAxisLocation',0)
%a=axes('xcolor',get(f,'color'),'xtick',[])
xlabel( '$x$' );
ylabel( '$\ell(x)$\qquad' );
set(get(gca,'XLabel'),'Rotation',0,'Interpreter','latex');
set(get(gca,'YLabel'),'Rotation',0,'Interpreter','latex');
set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off');
set(gcf, 'color', 'white');
set(gca, 'YGrid', 'off');




% Make a second set of axes for the right-hand side label.
%ah=axes('position',get(gca,'position'),...
%        'YAxisLocation','right',... 'Color','none',... 'visible','on');
%set( gca, 'XTick', [] ); set( gca, 'yTick', [] ); set( gca, 'XTickLabel',
%'' ); set( gca, 'yTickLabel', '' ); set(gca,'xColor','g');
%set(gca,'yColor','k'); set(
%get(gca,'YLabel'),'Rotation',0,'Interpreter','latex', 'Fontsize',
%fontsize); ylabel( '\quad$Z$' );


hLgnd = legend( [h_points, h_postmean, h_fill1, g(1), post_h, draw_h' ], ...
    {'samples', '\gpb mean', '\gpb mean $\pm$ \acro{sd}', 'expected $Z$', '$p(Z | \text{samples})$', 'draw from \gp', 'draw from \gp', 'draw from \gp'},...
    'Location', 'EastOutside');
legend boxoff


set_fig_units_cm( 24, 5 );
%set(hLgnd, 'Units','pixels'); pos = get(hLgnd,'Position'); legendpos =
%[pos(1), pos(2) - 10, pos(3), pos(4)]; set(hLgnd,'Position', legendpos);

%legendpos(1) = legendpos(1) + legendpos(3) * 2; hLgnd2 = legend( ], ...
%    {}, ... 'Location', 'EastOutside',
%    'Interpreter','latex');
%legend boxoff set(hLgnd2, 'Units','pixels'); set(hLgnd2,
%'OuterPosition',legendpos); set(hLgnd2, 'Units','normalized');

%set_fig_units_cm( 20, 8 );
matlabfrag('~/Docs/sbq-paper/figures/bmc_intro6');
%save2pdf('~/Dropbox/papers/infinite-bq/figures/bq_intro5.pdf', gcf, 300,
%true);
%saveeps('bq_intro5');
end


