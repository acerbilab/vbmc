% gp_plot (x, m, sd, obs_x, obs_y, real_x, real_y, params)
% x - x values where predictions are made
% m - mean, should be same length as x
% sd - standard deviation, sould be same length as x
% obs_x - x values of observations
% obs_y - y values of observations, should be same length as obs_x
% bounds - [min_x max_x min_y max_y] bounds for axes
% dot_size - size of observation dots
% x_label - x label text
% y_label - y label text
% obs_label - text to use for observation dots in legend
% legend_location - where to place legend (e.g. 'NorthEast')
% width - width in centimeters
% height - height in centimeters
% name - filename to save plot to
% legend_location2 (optional) - if you'd prefer to have two
%   legends, one for observations and the other for mean/sd, the
%   location of the mean/sd legend.

function [meanh, SDh] = gp_plot (x, m, sd, obs_x, obs_y, real_x, real_y, params)

if nargin<8
    params = struct();
    if nargin<7
        real_y = [];
        if nargin<6
            real_x = [];
        end
    end
end

dark_colours = cbrewer('qual','Set1', 5);
light_colours = cbrewer('qual','Pastel1', 5);

default_params = struct('bounds', false, ...
                       'dot_size', 7, ...
                       'x_label', 'x', ...
                       'y_label', 'y', ...
                       'obs_label','data', ...
                       'legend_location', 'Best', ...
                       'width', 15, ...
                       'height', 7, ...
                       'name', '', ...
                       'background', false, ...
                       'mean_line', true, ...
                       'colour', dark_colours(5,:), ...
                       'fill_colour', light_colours(5,:));
 
names = fieldnames(default_params);
for i = 1:length(names);
    name = names{i};
    if (~isfield(params, name))
        params.(name) = default_params.(name);
    end
end
 
x = x(:);
m = m(:);
sd = sd(:);

sd = real(sd);
sd(isnan(sd)) = 0;
sd(sd == 0) = eps;
m(isnan(m)) = 0;


[x,I] = sort(x);
m = m(I);
sd = sd(I);


real_x = real_x(:);
real_y = real_y(:);
[real_x,I] = sort(real_x);
real_y = real_y(I);


hold on;
SDh = fill([x; x(end:-1:1)], ...
  [m + 2 * sd; m(end:-1:1) - 2 * sd(end:-1:1)], ...
  params.fill_colour, 'EdgeColor', params.fill_colour);
meanh = plot(x, m);
realh = plot(real_x,real_y,'-');
observationsh = plot(obs_x, obs_y, '.');


fh = gcf;
fa = gca;

set(fh, 'units', 'centimeters', ... 
  'NumberTitle', 'off', 'Name', 'plot');
pos = get(fh, 'position'); 
set(fh, 'position', [pos(1:2), params.width, params.height]); 

if iscell(params.bounds)
    if ~isempty(params.bounds{1})
        xlim(params.bounds{1});
    end
    if ~isempty(params.bounds{2})
        ylim(params.bounds{2});
    end
elseif params.bounds
    axis(params.bounds);
end

if params.background
    xlims = get(gca, 'xlim');
    ylims = get(gca, 'ylim');

    sz_x = diff(xlims);
    sz_y = diff(ylims);

    background = imread('background.jpeg');
    [bg_x, bg_y, bg_z] = size(background);

    % This creates the 'background' axes
    ha = axes('units','normalized', ...
    'position',get(gca, 'position'));

    % Move the background axes to the bottom
    image(background);

    set(ha,'handlevisibility','off', ...
    'visible','off')

    axes(fa);
    set(fa, 'Color', 'none');
end


set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(fh, 'color', 'white'); 
set(gca, 'YGrid', 'off');
set(gca, 'color', 'white'); 


set(realh, ... 
  'LineStyle', '-', ...
  'LineWidth', 0.75, ...
  'Marker', 'none', ...
  'Color', [0.6 0.6 0.6] ...
  );

if params.mean_line
    set(meanh, ... 
      'LineStyle', '-', ...
      'LineWidth', 0.75, ...
      'Color', params.colour ...
      );
else
    set(meanh, ... 
      'LineStyle', 'none', ...
      'LineWidth', 0.75, ...
      'Color', params.colour ...
      );
end

set(observationsh, ... 
  'LineStyle', 'none', ...
  'LineWidth', 0.5, ...
  'Marker', '.', ...
  'MarkerSize', params.dot_size, ...
  'Color', [0.2 0.2 0.2] ...
  );

if ischar(params.x_label)
xlabel(params.x_label)
end

if length(params.y_label)>3
    Rotation = 90;
else
    Rotation = 0;
end
ylabel(params.y_label,'Rotation',Rotation)
title(params.name)

if ischar(params.legend_location)
if ~isempty(real_y)
    l =legend( ...
    [observationsh, meanh, SDh, realh], ...
    params.obs_label, ...
    'mean' , ...
    '$\pm 2$ SD', ...
    'true values', ...
    'Location', params.legend_location);
elseif ~params.mean_line
        l =legend( ...
    [observationsh, SDh, realh], ...
    params.obs_label, ...
    'mean $\pm 2$ SD', ...
    'true values', ...
    'Location', params.legend_location);
else
    l = legend( ...
    [observationsh, meanh, SDh], ...
    params.obs_label, ...
    'mean' , ...
    '$\pm 2$ SD', ...
    'Location', params.legend_location);
end

legend boxoff
end
set(0, 'defaulttextinterpreter', 'none')