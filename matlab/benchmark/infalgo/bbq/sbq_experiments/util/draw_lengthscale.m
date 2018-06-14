function draw_lengthscale( lengthscale, name, multiple )
%
% Draws a nice line to show a lengthscale.
%
% David Duvenaud
% February 2012

if nargin < 3; multiple = 1; end;
if nargin < 2; name = 'lengthscale'; end;

y_limits = ylim;
y_scale = y_limits(2) - y_limits(1);
yval1 = y_limits(1) +  y_scale * (0.1 * multiple + 0.05);
x_limits = xlim;
x_loc = x_limits(1) + 0.05*(x_limits(2) - x_limits(1));
line([x_loc, x_loc + lengthscale],[yval1,yval1], 'Color', 'k', 'Linewidth', 2);
line([x_loc, x_loc],[yval1 + 0.01*y_scale,yval1 - 0.01*y_scale], ...
     'Color', 'k', 'Linewidth', 2);
line([x_loc + lengthscale, x_loc + lengthscale],...
     [yval1 + 0.01*y_scale,yval1 - 0.01*y_scale], 'Color', 'k', 'Linewidth', 2);

yval2 = yval1 + 0.05 .* y_scale;
text( x_loc, yval2, name );
