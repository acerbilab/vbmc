
addpath(genpath('~/Code/convex_programming'))

colours = (cbrewer('qual','Set1',9));

N = 2;

% generate random covariance
% Sigma = rand(N)-0.5;
% Sigma = Sigma' * Sigma;

Sigma = [0.2035    0.0655
        0.0655    0.0879];

% found bound
cvx_begin sdp
variable L(N,N) diagonal
minimize(trace(L))
L >= Sigma
cvx_end

figure(2);clf;hold on;

f = linspace(-1,1,1000);
real_h = plot(f, lognormpdf(f, 0, sqrt(Sigma(1))), 'Color',colours(1,:));
bound_h = plot(f, lognormpdf(f, 0, sqrt(L(1))), 'Color',colours(2,:));

xlabel $x_1$
ylabel 'log density'
set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(gcf, 'color', 'white'); 
set(gca, 'YGrid', 'off');
set(gca, 'color', 'white'); 
set(gca, 'xtick', [])
set(gca, 'ytick', [])

axis tight
fh = gcf;

set(gca,'LooseInset',get(gca,'TightInset'))
h_legend = legend([real_h, bound_h], ...
    sprintf('$\\Sigma$'), ...
    sprintf('$\\Lambda$'), ...
    'Location','South');
legend boxoff

width = 4; height = 3.5;


pos = get(fh, 'position'); 
set(fh, 'units', 'centimeters', ... 
  'NumberTitle', 'off', 'Name', 'plot');
set(fh, 'position', [pos(1:2), width, height]); 

% pos = get(h_legend,'position');
% set(h_legend,'Units','centimeters');
% set(h_legend, 'position',[0 4 pos(3:4)]);



set(0, 'defaulttextinterpreter', 'none')
matlabfrag('~/Docs/bayes-quadrature-for-gaussian-integrals-paper/figures/bounding_pdf')


figure(3);clf;hold on;


real_h = error_ellipse(Sigma, 'lines', false, 'style',...
    {'Color',colours(1,:),'LineWidth', 1});
error_ellipse(Sigma, 'lines', true, 'style',...
    {'Color',colours(1,:),'LineWidth', 0.5});
bound_h = error_ellipse(L, 'lines', false, 'style',...
    {'Color',colours(2,:),'LineWidth', 1});
error_ellipse(L, 'lines', true, 'style',...
    {'Color',colours(2,:),'LineWidth', 0.5});





xlabel $x_1$
ylabel('$x_2$','Rotation', 0)
set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(gcf, 'color', 'white'); 
set(gca, 'YGrid', 'off');
set(gca, 'color', 'white'); 
set(gca, 'xtick', [])
set(gca, 'ytick', [])

axis square
fh = gcf;

width = 4; height = 3.5;


pos = get(fh, 'position'); 
set(fh, 'units', 'centimeters', ... 
  'NumberTitle', 'off', 'Name', 'plot');
set(fh, 'position', [pos(1:2), width, height]); 

% pos = get(h_legend,'position');
% set(h_legend,'Units','centimeters');
% set(h_legend, 'position',[0 4 pos(3:4)]);



set(0, 'defaulttextinterpreter', 'none')
matlabfrag('~/Docs/bayes-quadrature-for-gaussian-integrals-paper/figures/bounding_ellipse')
