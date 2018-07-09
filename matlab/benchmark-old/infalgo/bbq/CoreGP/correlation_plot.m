function [onetooneh,meanh,SDh] = correlation_plot(m, sd, real_y, params)
% [onetooneh,meanh,SDh] = correlation_plot(m, sd, real_y, params)
% m is the mean and sd the standard dev for real_y

if nargin<4
    params = struct();
end

default_params = struct('bounds', false, ...
                        'dot_size', 7, ...
                       'x_label', 'true value', ...
                       'y_label', 'predictions', ...
                       'obs_label','data', ...
                       'legend_location', 'Best', ...
                       'width', 12, ...
                       'height', 9, ...
                       'name', '',...
                       'step', false);
 
names = fieldnames(default_params);
for i = 1:length(names);
    name = names{i};
    if (~isfield(params, name))
        params.(name) = default_params.(name);
    end
end

fh = gcf;

set(fh, 'units', 'centimeters', ... 
  'NumberTitle', 'off', 'Name', 'plot');
pos = get(fh, 'position'); 
set(fh, 'position', [pos(1:2), params.width, params.height]); 


set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(fh, 'color', 'white'); 
set(gca, 'YGrid', 'off');

sd = real(sd);
sd(isnan(sd)) = 0;
sd(sd == 0) = eps;
m(isnan(m)) = 0;

[real_y, I] = sort(real_y);
m = m(I);
sd = sd(I);

hold on

if islogical(params.step)
    step = 0.5*mean(diff(real_y));
else
    step = params.step;
end

for i = 1:length(real_y)
SDh = fill([real_y(i)-step; real_y(i)+step; real_y(i)+step; real_y(i)-step], ...
  [m(i) + 2 * sd(i); m(i) + 2 * sd(i); m(i) - 2 * sd(i); m(i) - 2 * sd(i)], ...
  [0.87 0.89 1], 'EdgeColor', 'none');
end
meanh = plot(real_y, m,'.');

full_data = [m-2*sd;m+2*sd;real_y];
miny = min(full_data);
maxy = max(full_data);


if iscell(params.bounds)
    if ~isempty(params.bounds{1})
        xlim(params.bounds{1});
    end
    if ~isempty(params.bounds{2})
        ylim(params.bounds{2});
    end
elseif ~isempty(params.bounds) && ~islogical(params.bounds)
    axis(params.bounds);
else
    axis([miny maxy, miny maxy])
end
axis square

set(meanh, ... 
  'LineStyle', 'none', ...
  'LineWidth', 0.75, ...
  'Marker', '.', ...
  'MarkerSize', params.dot_size, ...
  'Color', [0 0 0.8] ...
  );

hold on
onetooneh = plot([miny,maxy], [miny,maxy],'k-');
set(onetooneh, ... 
  'LineStyle', '-', ...
  'LineWidth', 0.75, ...
  'Color', [0 0 0] ...
  );

xlabel(params.x_label)

if length(params.y_label)>3
    Rotation = 90;
else
    Rotation = 0;
end
ylabel(params.y_label,'Rotation',Rotation)
title(params.name)

if ~islogical(params.legend_location)
l =legend( ...
[onetooneh,meanh,SDh], ...
'1:1', ...
    'mean' , ...
    ['$\pm 2$',sprintf('\n'),'SD'], ...
'Location', params.legend_location);
end

legend boxoff