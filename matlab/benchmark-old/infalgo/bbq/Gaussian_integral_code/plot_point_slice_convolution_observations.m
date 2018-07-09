figure(1)
clf

% R = rand(2);
% Sigma = R' * R;

Sigma = [ 0.9228    0.2679
    0.2679    0.1611];

N = 100;
x_range = linspace(-2, 2, N);
y_range = x_range;

l_x = -1;
u_x = 1;
l_y = -1.5;
u_y = 1.5;
r = sqrt((u_x - l_x)^2 + (u_y - l_y)^2)/2;

cx_range = linspace(l_x, u_x, N);
cy_range = linspace(l_y, u_y , N);

[X, Y] = meshgrid(x_range, y_range);
[cX, cY] = meshgrid(cx_range, cy_range);


P = nan(N);
cP = nan(N);

for  i = 1:N
    for j = 1:N
        P(i, j) = mvnpdf([X(i, j);Y(i, j)], [0;0], Sigma);
        cP(i, j) = mvnpdf([cX(i, j);cY(i, j)], [0;0], Sigma);
    end
end

% ====
[undC, undesired_h] = contour(X, Y, P, 5, 'k');
hold on

% ====
colours = cbrewer('seq','YlGnBu', 9);
colormap(colours);
[dC, desired_h] = contourf(cX, cY, cP);
shading flat

rectangle('Position', [l_x l_y u_x-l_x u_y-l_y],'EdgeColor', 'k');
% ====
%S = mvnrnd(zeros(10,2), Sigma);
S = [-0.1899    0.1474
   -1.1603   -0.9291
    2.7935    0.7088
    0.7927   -0.0076
    1.3247   -0.0707
   -1.0165   -0.1485
   -0.4502   -0.0493
   -0.2617   -0.0663
    1.0552   -0.0787
   -0.2669    0.2480];

plot(S(:, 1), S(:, 2), 'w.','MarkerSize', 15);
pts_h = plot(S(:, 1), S(:, 2), 'k.','MarkerSize', 9);

% % ====
% 
% plot(x_range, 1.2 * ones(N, 1), 'k', 'LineWidth', 2);
% slice_h = plot(x_range, -1.2 * ones(N, 1),'k', 'LineWidth', 2);

% ====

% S = rand(2);
% V = S' * S;

V = diag([0.4 2]);

colours = cbrewer('qual','Set1',9);

conv_h = error_ellipse(V, [(l_x+u_x)/2;0], 'style',...
    {'Color',colours(8,:),'LineWidth', 1}, 'clip_bounds', [-inf, inf; l_y, u_y]);

V = diag([1 2.5]);

conv2_h = error_ellipse(V, [(l_x+u_x)/2;0], 'style',...
    {'Color',colours(1,:),'LineWidth', 1}, 'clip_bounds', [l_x, u_x; -inf, inf]);

% ====
set(gca,'LooseInset',get(gca,'TightInset'))
h_legend = legend([undesired_h, desired_h, pts_h, conv_h, conv2_h], ...
    '$\N{\vx}{\v{\mu}}{\Sigma}$', ...
    'integrand', ...
    sprintf('point\nobservations'), ...
    sprintf('slice\nobservation'),...
    sprintf('slice\nobservation'),...
    'Location','EastOutside');
legend boxoff

xlabel $x_1$
ylabel('$x_2$','Rotation', 0)
set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(gcf, 'color', 'white'); 
set(gca, 'YGrid', 'off');
set(gca, 'color', 'white'); 
set(gca, 'xtick', [l_x, u_x])
set(gca, 'ytick', [l_y, u_y])
set(gca, 'xticklabel', {'$l_1$', '$u_1$'});
set(gca, 'yticklabel', {'$l_2$', '$u_2$'});

set(get(gca,'YLabel'),'Position',[min(x_range) max(y_range)])
set(get(gca,'XLabel'),'Position',[max(x_range)+0.3 min(y_range)+0.2])

fh = gcf;

width = 10.7; height = 4;


pos = get(fh, 'position'); 
set(fh, 'units', 'centimeters', ... 
  'NumberTitle', 'off', 'Name', 'plot');
set(fh, 'position', [pos(1:2), width, height]); 

% pos = get(h_legend,'position');
% set(h_legend,'Units','centimeters');
% set(h_legend, 'position',[0 4 pos(3:4)]);



set(0, 'defaulttextinterpreter', 'none')
matlabfrag('~/Docs/bayes-quadrature-for-gaussian-integrals-paper/figures/observations')
