function [h,g] = crosshatch_poly(x, y, lineangle, linegap, varargin)
% Fill a convex polygon with regular diagonal lines (hatching, or cross-hatching)
%
% file:      	crosshatch_poly.m, (c) Matthew Roughan, Mon Jul 20 2009
% created: 	Mon Jul 20 2009 
% author:  	Matthew Roughan 
% email:   	matthew.roughan@adelaide.edu.au
% 
%
% crosshatch_poly(x, y, angle, gap) fills the 2-D polygon defined by vectors x and y
% with slanted lines at the specified lineangle and linegap. 
%
% The one major limitation at present is that the polygon must be convex -- if it is not, the
% function will fill the convex hull of the supplied vertices. Non-convex polygons must be
% broken into convex chunks, which is a big limitation at present.
%
% Cross-hatching can be easily achieved by calling the function twice.
%
% Speckling can be roughly achieved using linestyle ':'
% 
%
% INPUTS:
%     (x,y)  gives a series of points that defines a convex polygon.   
%     lineangle  the angle of the lines used to fill the polygon
%            specified in degrees with respect to vertical
%            default = 45 degrees
%     linegap    the gap between the lines used to fill the polygon
%            default = 1
%     options can be specified in standard Matlab (name, value) pairs
%        'edgecolor'  color of the boundary line of the polygon
%        'edgewidth'  width of the boundary line of the polygon
%                         0 means no line
%        'edgestyle'  style of the boundary line of the polygon
%        'linecolor'  color of the fill lines
%        'linewidth'  width of fill lines
%        'linestyle'  style of fill lines
%        'backgroundcolor'  background colour to fill the polygon with
%                     if not specified, no fill will be done
%        'hold_n'     hold_n=1 means keep previous plot
%                     hold_n=0 (default) means clear previous figure
%
% OUTPUTS:
%    h = a vector of handles to the edges of the polygon
%    g = a vector of handles to the lines
%
% Works by finding intersection points of the fill lines with the boundary lines, and then
% drawing a line between intersection points that lie on the boundary of the polygon.
%
% version 0.1, Jul 20th 2009, Matthew Roughan
% version 0.2, Jul 22nd 2009, fixed typo, Matthew Roughan
%
%
% There are a number of similar functions, that I'll point to, but they are a little
% different as well.
%    linpat.m by Stefan Bilig does essentially the same thing, but only in rectangular regions
%    applyhatch_pluscolor.m by Brandon Levey (from Brian Katz and Ben Hilig) maps colors in
%             an image to patterns, which is cool, but I just want hatching to be easy, and
%             direct, so I can do things like plot two regions and cross hatch both
%    hatching.m by  ijtihadi ijtihadi does hatching between two (arbitrary) functions, which
%             could include many shapes, but isn't easy to use directly for polygons or other
%             shapes. Note that often smooth curves can be well approximated by polygons so
%             this function can be used for these cases as well.
% 
%
% TODO: speckles and other more interesting patterns
%       cross-hatching as a built in
%       avoid dependence on optimization toolbox
% 

% read the input options and set defaults
if (nargin < 4)
  linegap = 1;
end
if (nargin < 3)
  lineangle = 45;
end
edgecolor = 'k'; 
edgewidth = 1;
edgestyle = '-';
linecolor = 'k'; 
linewidth = 1;
linestyle = '-';
hold_n = 0;
if (length(varargin) > 0)
  % process variable arguments
  for k = 1:2:length(varargin)
    if (ischar(varargin{k}))
      argument = char(varargin{k}); 
      value = varargin{k+1};
      switch argument
       case {'linecolor','lc'}
	linecolor = value;
       case {'backgroundcolor','bgc'}
	backgroundcolor = value;
       case {'linewidth','lw'}
	linewidth = value;
       case {'linestyle','ls'}
	linestyle = value;
       case {'edgecolor','ec'}
	edgecolor = value;
       case {'edgewidth','ew'}
	edgewidth = value;
       case {'edgestyle','ew'}
	edgestyle = value;
       case {'hold'}
	hold_n = value;
       otherwise
	error('incorrect input parameters');
      end
    end
  end
end

% reset plot of needed
if (hold_n==0)
  hold off
  plot(x(1), y(2));
end
hold on

% get the convex hull of the supplied vertices, partly to ensure convexity, but also to sort
% them into a sensible order
[k] = convhull(x,y);
x = x(k(1:end-1));
y = y(k(1:end-1));
N = length(k)-1;
% make everything row vectors
if (size(x,1) > 1)
  x = x';
end
if (size(y,1) > 1)
  y = y';
end

% if the background is set, then fill, and set the edge correctly
if (exist('backgroundcolor', 'var'))
  h = fill(x,y,backgroundcolor);
  if (edgewidth > 0)
    set(h, 'LineWidth', edgewidth, 'EdgeColor', edgecolor, 'LineStyle', edgestyle);
  else
    set(h, 'EdgeColor', backgroundcolor);
  end
end

% plot edges if needed
for i=1:N
  if (edgewidth > 0 & ~exist('backgroundcolor', 'var'))
    % only need to draw edges if width is > 0 and haven't already done so with fill
    j = mod(i, N)+1;
    h(i) = plot([x(i) x(j)], [y(i) y(j)], ...
		'color', edgecolor, 'linestyle', edgestyle, 'linewidth', edgewidth);
  end
end

% now find the range for the lines to plot
c = [cosd(lineangle), sind(lineangle)];  % normal to the lines
v = [sind(lineangle), -cosd(lineangle)]; % direction of lines
obj = c * [x; y];
[mx, kmx] = max(obj, [], 2);
[mn, kmn] = min(obj, [], 2);
% plot(x(kmx), y(kmx), 'r*');
% plot(x(kmn), y(kmn), 'ro');
distance = sqrt( (x(kmx)-x(kmn)).^2 + (y(kmx)-y(kmn)).^2  );

% find a line describing each edge
for i=1:N
  j = mod(i, N)+1;
  if (abs(x(j) - x(i)) > 1.0e-12)
    % find the slope and intersept
    slope(i) = (y(j) - y(i)) / (x(j) - x(i));
    y_int(i) = y(i) - slope(i)*x(i);
  else
    % the line is vertical
    slope(i) = Inf;
    y_int(i) = NaN;
  end
end

% now draw lines clipping them at points that are on the edge of the polygon
g = [];
% find a slightly larger polygon
centroid_x = mean(x);
centroid_y = mean(y);
epsilon = 0.001;
x_dash = (1+epsilon) * (x - centroid_x) + centroid_x;
y_dash = (1+epsilon) * (y - centroid_y) + centroid_y;
% fill(x_dash, y_dash, 'g');
for m=0:linegap:distance
  counter = ceil(m/linegap)+1;
  sigma = [x(kmn), y(kmn)] + m*c;
  % plot(sigma(1), sigma(2), '+', 'color', linecolor);
  
  % for each line, look where it intersepts the edge of polygon (if it does)
  for i=1:N
    % find the intercept with this line, and the relevant edge
    if (isinf(slope(i)))
      if (abs(v(1)) > 1.0e-12)
	t = (x(i) - sigma(1)) / v(1);
	x_i(i) = x(i);
	y_i(i) = sigma(2) + t * v(2);
      else
	x_i(i) = NaN;
	y_i(i) = NaN;
      end
    else
      if (abs(v(2) - slope(i)*v(1)) > 1.0e-12)
	t = (slope(i) * sigma(1) - sigma(2) + y_int(i)) / ( v(2) - slope(i)*v(1));
	x_i(i) = sigma(1) + t * v(1);
	y_i(i) = sigma(2) + t * v(2);
      else
	x_i(i) = NaN;
	y_i(i) = NaN;
      end
    end
  end
  k = find(inpolygon(x_i, y_i, x_dash, y_dash));
  if (length(k) == 2)
    g(counter) = plot(x_i(k), y_i(k), ...
		'color', linecolor, 'linestyle', linestyle, 'linewidth', linewidth);
  elseif (length(k) < 2)
    % don't plot because we have no clear line
  elseif (length(k) > 2)
    % find two unique points
    d = [x_i(k)', y_i(k)'];
    d = round(100*d)/100;
    d = unique(d, 'rows');
    g(counter) = plot(d(:,1), d(:,2), ...
		'color', linecolor, 'linestyle', linestyle, 'linewidth', linewidth);
  end
end
