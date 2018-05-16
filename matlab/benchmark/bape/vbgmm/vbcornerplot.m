function [fig,ax]=vbcornerplot(data, varargin)
%VBCORNERPLOT Corner plot showing projections of a multidimensional data set.
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
% Inspired by triangle.py (github.com/dfm/triangle.py)
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
vbmodel = [];
bounds_supplied = true;
top_margin = 0;

if nargin > 1
    names = varargin{1};
    if ~isempty(names) && ~(iscell(names) && length(names) == nDims)
        error('NAMES must be a cell array with length equal to the number of dimensions in your data.')
    end
    if nargin > 2
        truths = varargin{2};
        if ~isempty(truths) && ~(isfloat(truths) && size(truths,2) == nDims)
            error('TRUTHS must be a row vector of matrix with columns equal to the number of dimensions in your data.')
        end
        if nargin > 3
            bounds = varargin{3};
            
            if ~isempty(bounds) && ~(isfloat(bounds) && all(size(bounds) == [2 nDims]))
                error('BOUNDS must be a 2-by-nDims matrix.')
            end
            if nargin > 4
                if ~isempty(varargin{4}); top_margin = varargin{4}; end
                
                if nargin > 5
                    vbmodel = varargin{5};
                end                
            end
        end
    end
end

if isempty(bounds)
    bounds = nan(2,nDims);
    bounds_supplied = false;
end

% plotting parameters
fig = figure;
ax = nan(nDims+top_margin,nDims);
hist_bins = 40;
lines = 10;
res = 2^6; % defines grid for which kde2d will compute density. must be a power of 2.
linewidth = 1;
axes_defaults = struct('tickdirmode','manual',...
    'tickdir','out',...
    'ticklength',[.035 .035],...
    'box','off',...
    'color','none',...
    'xticklabel',[],...
    'yticklabel',[]);

% plot histograms
for i = 1:nDims
    if ~bounds_supplied
        bounds(:,i) = [min(data(:,i)) max(data(:,i))];
        % bounds(:,i) = [prctile(data(:,i),20), prctile(data(:,i),80)];
    end
    truncated_data = data;
    truncated_data(truncated_data(:,i)<bounds(1,i) | truncated_data(:,i)>bounds(2,i),i) = nan;

    ax(i,i) = tight_subplot(2+nDims, 1+nDims, i, i+1);
    set(gca,'visible','off','xlim',bounds(:,i))
    ax(i+top_margin,i) = tight_subplot(1+top_margin+nDims,1+nDims, i+top_margin, i+1);
    
    [n,x] = hist(truncated_data(:,i), hist_bins);
    plot(x,n/sum(n),'k-');
    dx = mean(diff(x));
    set(gca,'xlim',bounds(:,i),'ylim',[0 max(n/sum(n))],axes_defaults,'ytick',[]);
    
    if i == nDims
        set(gca,'xticklabelmode','auto')
    end
    
    if ~isempty(truths)
        hold on
        for l = 1:size(truths,1)
            plot([truths(l,i) truths(l,i)], [0 1], 'k-', 'linewidth',linewidth)
        end
    end
    
    if ~isempty(names)
        if i == 1
            ylabel(names{i});
        end
        if i == nDims
            xlabel(names{i})
        end
    end
    
    if ~isempty(vbmodel)
        hold on;
        vb1 = vbgmmmarg(vbmodel,i);
        xx = linspace(bounds(1,i),bounds(2,i),1e4);
        dxx = xx(2)-xx(1);
        y = vbgmmpred(vb1,xx);
        y(isnan(y)) = 0;
        plot(xx,y/sum(y)*dx/dxx,'k--','linewidth',linewidth);
    end
    
end

% plot projections
if nDims > 1
    for d1 = 1:nDims-1 % col
        for d2 = d1+1:nDims % row
            [~, density, X, Y] = kde2d([data(:,d1) data(:,d2)],res,[bounds(1,d1) bounds(1,d2)],[bounds(2,d1) bounds(2,d2)]);

            ax(d2+top_margin,d1) = tight_subplot(top_margin+1+nDims,1+nDims, d2+top_margin, 1+d1);
            contour(X,Y,density, lines)
            
            set(gca,'xlim',bounds(:,d1),'ylim',bounds(:,d2), axes_defaults);
            
            if ~isempty(truths)
                yl = get(gca,'ylim');
                xl = get(gca,'xlim');
                hold on
                for l = 1:size(truths,1)
                    plot(xl, [truths(l,d2) truths(l,d2)],'k-', 'linewidth',linewidth)
                    plot([truths(l,d1) truths(l,d1)], yl,'k-', 'linewidth',linewidth)
                end
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

set(gcf,'Color','w');

end

function h=tight_subplot(m,n,row,col)
% adapted from subplot_tight
% (mathworks.com/matlabcentral/fileexchange/30884-controllable-tight-subplot/)
% by Nikolay S. (http://vision.technion.ac.il/~kolian1/)

gutter = [.015, .015];

height = (1-(m+1)*gutter(1))/m;% plot height
width = (1-(n+1)*gutter(2))/n; % plot width
bottom = (m-row)*(height+gutter(1))+gutter(1); % bottom pos
left = col*(width+gutter(2))-width;            % left pos

pos_vec= [left bottom width height];

h=subplot('Position',pos_vec);
end