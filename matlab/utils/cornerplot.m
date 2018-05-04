function [fig,ax]=cornerplot(data, varargin)
%CORNERPLOT Corner plot showing projections of a multidimensional data set.
%
% CORNERPLOT(DATA) plots every 2D projection of a multidimensional data
% set. DATA is an nSamples-by-nDimensions matrix.
%
% CORNERPLOT(DATA,NAMES) prints the names of each dimension. NAMES is a
% cell array of strings of length nDimensions, or an empty cell array.
%
% CORNERPLOT(DATA,NAMES,TRUTHS) indicates reference values on the plots.
% TRUTHS is a vector of length nDimensions. This might be useful, for instance,
% when looking at samples from fitting to synthetic data, where true
% parameter values are known.
%
% CORNERPLOT(DATA,NAMES,TRUTHS,BOUNDS) indicates lower and upper bounds for
% each dimension. BOUNDS is a 2 by nDimensions matrix where the first row is the
% lower bound for each dimension, and the second row is the upper bound.
%
% CORNERPLOT(DATA,NAMES,TRUTHS,BOUNDS,TOP_MARGIN) plots a number, defined by TOP_MARGIN,
% of empty axes at the top of each column. This can be useful for plotting other statistics
% across parameter values (eg, marginal log likelihood).
%
%
% Output:
% FIG is the handle for the figure, and AX is a
% nDimensions-by-nDimensions array of all subplot handles.
%
% Inspired by corner (github.com/dfm/corner.py)
% by Dan Foreman-Mackey (dan.iel.fm).
%
% Requires kde2d
% (mathworks.com/matlabcentral/fileexchange/17204-kernel-density-estimation/content/kde2d.m)
% by Zdravko Botev (web.maths.unsw.edu.au/~zdravkobotev/).
%
% William Adler, January 2015
% Ver 1.0
% will@wtadler.com

if ~exist('kde2d','file')
    error('You must install <a href="http://www.mathworks.com/matlabcentral/fileexchange/17204-kernel-density-estimation/content/kde2d.m">kde2d.m</a> by <a href="http://web.maths.unsw.edu.au/~zdravkobotev/">Zdravko Botev</a>.')
end

if length(size(data)) ~= 2
    error('x must be 2D.')
end

nDims = min(size(data));  % this assumes that you have more samples than dimensions in your data. Hopefully this is a safe assumption!

% make sure columns are the dimensions of the data
if nDims ~= size(data,2)
    data = data';
end

% assign names and truths if given
names = {};
truths = [];
bounds = [];
bounds_supplied = true;
top_margin = 0;

gutter = [.004 .004];
margins = [.1 .01 .12 .01];

if nargin > 1
    names = varargin{1};
    if ~isempty(names) && ~(iscell(names) && length(names) == nDims)
        error('NAMES must be a cell array with length equal to the number of dimensions in your data.')
    end
    if nargin > 2
        truths = varargin{2};
        if ~isempty(truths) && ~(isfloat(truths) && numel(truths) == nDims)
            error('TRUTHS must be a vector with length equal to the number of dimensions in your data.')
        end
        if nargin > 3
            bounds = varargin{3};
            
            if ~isempty(bounds) && ~(isfloat(bounds) && all(size(bounds) == [2 nDims]))
                error('BOUNDS must be a 2-by-nDims matrix.')
            end
            if nargin > 4
                top_margin = varargin{4};
            end
        end
    end
end

if isempty(bounds) | all(bounds==0)
    bounds = nan(2,nDims);
    bounds_supplied = false;
end

% plotting parameters
fig = figure;
set(gcf, 'color', 'w')
ax = nan(nDims+top_margin,nDims);
hist_bins = 40;
lines = 10;
res = 2^6; % defines grid for which kde2d will compute density. must be a power of 2.
linewidth = 1;
axes_defaults = struct('tickdirmode','manual',...
    'tickdir','out',...
    'ticklength',[.035 .035],...
    'box','off',...
    'xticklabel',[],...
    'yticklabel',[]);

% plot histograms
for i = 1:nDims
    if ~bounds_supplied
        bounds(:,i) = [min(data(:,i)) max(data(:,i))];
    end
    
    for t = 1:top_margin
        ax(i-1+t,i) = tight_subplot(nDims+top_margin, nDims, i-1+t, i, gutter, margins);
        set(gca,'visible','off','xlim',bounds(:,i));
    end

    truncated_data = data;
    truncated_data(truncated_data(:,i)<bounds(1,i) | truncated_data(:,i)>bounds(2,i),i) = nan;
    
    ax(i+top_margin,i) = tight_subplot(nDims+top_margin, nDims, i+top_margin, i, gutter, margins);
    
    h=histogram(truncated_data(:,i), hist_bins, 'normalization', 'probability', 'displaystyle', 'stairs', 'edgecolor', 'k');
    set(gca,'xlim',bounds(:,i),'ylim', [0 max(h.Values)], axes_defaults,'ytick',[]);
    
    if i == nDims
        set(gca,'xticklabelmode','auto')
    end
    
    if ~isempty(truths)
        hold on
        plot([truths(i) truths(i)], [0 1], 'k-', 'linewidth',linewidth)
    end
    
    if ~isempty(names)
        if i == 1
            ylabel(names{i});
        end
        if i == nDims
            xlabel(names{i})
        end
    end
    
end

% plot projections
if nDims > 1
    for d1 = 1:nDims-1 % col
        for d2 = d1+1:nDims % row
            [~, density, X, Y] = kde2d([data(:,d1) data(:,d2)],res,[bounds(1,d1) bounds(1,d2)],[bounds(2,d1) bounds(2,d2)]);

            ax(d2+top_margin,d1) = tight_subplot(nDims+top_margin, nDims, d2+top_margin, d1, gutter, margins);
            contour(X,Y,density, lines)
            
            set(gca,'xlim',bounds(:,d1),'ylim',bounds(:,d2), axes_defaults);
            
            if ~isempty(truths)
                yl = get(gca,'ylim');
                xl = get(gca,'xlim');
                hold on
                plot(xl, [truths(d2) truths(d2)],'k-', 'linewidth',linewidth)
                plot([truths(d1) truths(d1)], yl,'k-', 'linewidth',linewidth)
            end
            if d1 == 1
                if ~isempty(names)
                    ylabel(names{d2})
                end
                set(gca,'yticklabelmode','auto')
            end
            if d2 == nDims
                if ~isempty(names)
                    xlabel(names{d1})
                end
                set(gca,'xticklabelmode','auto')
            end
        end
        
%         % link axes
        row = ax(1+top_margin+d1,:);
        row = row(~isnan(row));
        row = row(1:d1);
        
        col = ax(:,d1);
        col = col(~isnan(col));
        col = col(1:end);
        
        linkaxes(row, 'y');
        linkaxes(col, 'x');
        
    end
end
end

function h=tight_subplot(m, n, row, col, gutter, margins, varargin)
%TIGHT_SUBPLOT Replacement for SUBPLOT. Easier to specify size of grid, row/col, gutter, and margins
%
% TIGHT_SUBPLOT(M, N, ROW, COL) places a subplot on an M by N grid, at a
% specified ROW and COL. ROW and COL can also be ranges
%
% TIGHT_SUBPLOT(M, N, ROW, COL, GUTTER=.002) indicates the width of the spacing
% between subplots, in terms of proportion of the figure size. If GUTTER is
% a 2-length vector, the first number specifies the width of the spacing
% between columns, and the second number specifies the width of the spacing
% between rows. If GUTTER is a scalar, it specifies both widths. For
% instance, GUTTER = .05 will make each gutter equal to 5% of the figure
% width or height.
%
% TIGHT_SUBPLOT(M, N, ROW, COL, GUTTER=.002, MARGINS=[.06 .01 .04 .04]) indicates the margin on
% all four sides of the subplots. MARGINS = [LEFT RIGHT BOTTOM TOP]. This
% allows room for titles, labels, etc.
%
% Will Adler 2015
% will@wtadler.com

if nargin<5 || isempty(gutter)
    gutter = [.002, .002]; %horizontal, vertical
end

if length(gutter)==1
    gutter(2)=gutter;
elseif length(gutter) > 2
    error('GUTTER must be of length 1 or 2')
end

if nargin<6 || isempty(margins)
    margins = [.06 .01 .04 .04]; % L R B T
end

Lmargin = margins(1);
Rmargin = margins(2);
Bmargin = margins(3);
Tmargin = margins(4);

unit_height = (1-Bmargin-Tmargin-(m-1)*gutter(2))/m;
height = length(row)*unit_height + (length(row)-1)*gutter(2);

unit_width = (1-Lmargin-Rmargin-(n-1)*gutter(1))/n;
width = length(col)*unit_width + (length(col)-1)*gutter(1);

bottom = (m-max(row))*(unit_height+gutter(2))+Bmargin;
left   = (min(col)-1)*(unit_width +gutter(1))+Lmargin;

pos_vec= [left bottom width height];

h=subplot('Position', pos_vec, varargin{:});
end