 clear all, close all;
 
 s = 1;
 m = 0;

 v = linspace(-3,3,100);
 [X, Y] = meshgrid(v, v);
 
 % real covariance:
 Kr = normpdf(X, m, s) .* normpdf(Y, m, s);
 
 % approximate covariance:
 Ka = normpdf(X, m, s) .* normpdf(Y, m, s) .* normpdf(X - Y, 0, s);
 
  
colours = flipud(colorbrew(1:11, 'divb', 11));
colormap(colours);

subplot(1,2,1)

contourf(X, Y, Kr);

title('real covariance')
  
set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(gcf, 'color', 'white'); 
set(gca, 'YGrid', 'off');
set(gca, 'color', 'white'); 
xlabel x_1
ylabel('x_2', 'Rotation',0);

%colorbar

subplot(1,2,2)

contourf(X, Y, Ka);
  
set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(gcf, 'color', 'white'); 
set(gca, 'YGrid', 'off');
set(gca, 'color', 'white'); 
xlabel x_1
ylabel('x_2', 'Rotation',0);

title('approx covariance')

%colorbar
