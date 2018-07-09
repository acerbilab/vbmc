function summary = benchmark_summaryplot(benchdata,fieldname,summary,options)
%BENCHMARK_SUMMARYPLOT Display summary optimization benchmark results.
%
%   See also BENCHMARK_PLOT.

% Luigi Acerbi 2016

if nargin < 2; fieldname = []; end
if nargin < 3; summary = []; end
if nargin < 4; options = []; end

% Get options and remove field
if isfield(benchdata,'options') && isempty(options)
    options = benchdata.options;
    benchdata = rmfield(benchdata,'options');
end

nx = 201;

ff = fields(benchdata)';

if any(strcmp(ff,'yy'))
    
    switch lower(options.Method)
        case 'ert'
            xrange = exp(linspace(log(10),log(1e3),nx));
            xx = benchdata.xx;
            yy = interp1(xx,benchdata.yy,xrange);
            xtick = [1e-2,1e-1,1];
            xticklabel = {'5·D','50·D','500·D'};
        case {'ir','fs'}
            % xrange = logspace(-2,0,200);
            xrange = exp(linspace(log(10),log(500),nx));
            xx = benchdata.xx;
            yy = interp1(xx,benchdata.yy,xrange);
            xtick = [1,1e1,1e2,1e3];
            xticklabel = {'1','10','100','1000'};
        case {'fst'}
            xrange = exp(linspace(log(max(benchdata.xx)),log(min(benchdata.xx)),nx));
            xx = benchdata.xx;
            yy = interp1(xx,benchdata.yy,xrange);
            xtick = [0.1,1,10];
            xticklabel = {'0.1','1','10'};
            
            
    end
    
    MinFval = NaN;
    if isfield(benchdata,'MinFval'); MinFval = benchdata.MinFval; end
    
    if isfield(summary,fieldname)
        summary.(fieldname).yy = yy + summary.(fieldname).yy;
        summary.(fieldname).n = summary.(fieldname).n + 1;
        summary.(fieldname).MinFval = min(summary.(fieldname).MinFval,MinFval);
        if isfield(benchdata, 'Zscores')            
            summary.(fieldname).Zscores = [summary.(fieldname).Zscores; benchdata.Zscores(:)];
            summary.(fieldname).Errs = [summary.(fieldname).Errs; benchdata.Errs(:)];
        end
    else
        summary.(fieldname).xrange = xrange;
        summary.(fieldname).yy = yy;
        summary.(fieldname).n = 1;
        summary.(fieldname).MinFval = MinFval;
        if isfield(benchdata, 'Zscores')
            summary.(fieldname).Zscores = benchdata.Zscores(:);
            summary.(fieldname).Errs = benchdata.Errs(:);
        end
    end
    
else
    for f = ff
        if isstruct(benchdata.(f{:}))
            summary = benchmark_summaryplot(benchdata.(f{:}),f{:},summary,options);
        end
    end
end

if ~isempty(fieldname); return; end

defaults = benchmark_defaults('options');
linstyle = defaults.LineStyle;
lincol = defaults.LineColor;
markersize = 10;
markerlinewidth = 1;
reverseX_flag = 0;

% List of all tested methods
ff = fields(summary)';

xlims = [Inf,-Inf];
for iField = 1:numel(ff)
    f = ff{iField};
    hold on;    
    xx = summary.(f).xrange;
    xlims = [min(xlims(1),min(xx)*0.999),max(xlims(2),max(xx)*1.001)];
    yy = summary.(f).yy./summary.(f).n;
    xx = xx(~isnan(yy));
    yy = yy(~isnan(yy));
    
    style(iField) = benchmark_defaults('style',[],[],f);

%     enhance = enhanceline(numel(ff),options);    
%     if iField == enhance; lw = 4; else lw = 2; end
    plot(xx,yy,style(iField).linestyle,'LineWidth',style(iField).linewidth,'Color',style(iField).color); hold on;
    if ~isempty(style(iField).marker)
        plot(xx(1:5:end),yy(1:5:end),style(iField).marker,'MarkerSize',markersize,'LineWidth',markerlinewidth,'Color',style(iField).color);
    end
    MinFval(iField) = summary.(f).MinFval;
    fieldname{iField} = f;
    
    % Get performance
    if strcmpi(options.Method,'fs')
        perf(iField) = mean(yy);
%        perf(iField) = yy(end);
    else
        perf(iField) = mean(yy);
    end
end

NumZero = options.NumZero;
% xlims = [min(xrange,[],2) max(xrange,[],2)];
% set(gca,'Xlim',xlims,'XTick',xtick,'XTickLabel',xticklabel)

switch lower(options.Method)
    case 'ir'
        ylims = [NumZero,1e3];
        if NumZero < 1e-5
            ytick = [NumZero,1e-5,0.1,1,10,1e5,1e10];
            yticklabel = {'0','10^{-5}','0.1','1','10','10^5','10^{10}'};
        else
            ytick = [NumZero,0.1,1,10,1e3];                    
            yticklabel = {'10^{-3}','0.1','1','10','10^3'};
        end
        liney = [1 1];
        set(gca,'Ylim',ylims,'YTick',ytick,'YTickLabel',yticklabel);
        set(gca,'TickDir','out','Xscale','log','Yscale','log','TickLength',3*get(gca,'TickLength'));

    case {'fs'}
        ylims = [0,1];
        ytick = [0,0.25,0.5,0.75,1];
        yticklabel = {'0','0.25','0.5','0.75','1'};
        liney = [1 1];
        xtick = [10:10:100,200:100:1000];
        for i = 1:numel(xtick); xticklabel{i} = ''; end
        xticklabel{1} = '10';
        xticklabel{5} = '50';
        xticklabel{10} = '100';
        xticklabel{14} = '500';
        xticklabel{19} = '1000';
        set(gca,'TickDir','out','Xscale','log');
        set(gca,'Xlim',xlims,'XTick',xtick,'XTickLabel',xticklabel);
        set(gca,'Ylim',ylims,'YTick',ytick,'YTickLabel',yticklabel);
        % set(gca,'TickDir','out','Xscale','log','TickLength',3*get(gca,'TickLength'));
        set(gca,'TickDir','out','Xscale','log');
        grid on;
        
    case {'fst'}
        ylims = [0,1];
        ytick = [0,0.25,0.5,0.75,1];
        yticklabel = {'0','0.25','0.5','0.75','1'};
        liney = [1 1];
        % xtick = [0.01 0.03 0.1 0.3 1 3 10];
%        for i = 1:numel(xtick); xticklabel{i} = num2str(xtick(i)); end
        xtick = [0.01:0.01:0.09 0.1:0.1:0.9 1:1:10];
        for i = 1:numel(xtick); xticklabel{i} = ''; end
        xticklabel{1} = '0.01'; xticklabel{3} = '0.03'; xticklabel{10} = '0.1'; xticklabel{12} = '0.3'; xticklabel{19} = '1'; xticklabel{21} = '3'; xticklabel{28} = '10';
        set(gca,'TickDir','out','Xscale','log');
        set(gca,'Xlim',xlims,'XTick',xtick,'XTickLabel',xticklabel);
        set(gca,'Ylim',ylims,'YTick',ytick,'YTickLabel',yticklabel);
        % set(gca,'TickDir','out','Xscale','log','TickLength',3*get(gca,'TickLength'));
        set(gca,'TickDir','out','Xscale','log');
        reverseX_flag = 1;
        grid on;
        
        
    case 'ert'
        ylims = [0,1];
        ytick = [0,0.5,1];
        yticklabel = {'0','0.5','1'};
        liney = [1 1];
        xtick = [10:10:100,200:100:1000];
        for i = 1:numel(xtick); xticklabel{i} = ''; end
        xticklabel{1} = '10';
        xticklabel{5} = '50';
        xticklabel{10} = '100';
        xticklabel{14} = '500';
        xticklabel{19} = '1000';
        set(gca,'TickDir','out','Xscale','log');
        set(gca,'Xlim',xlims,'XTick',xtick,'XTickLabel',xticklabel);
        set(gca,'Ylim',ylims,'YTick',ytick,'YTickLabel',yticklabel);
        % set(gca,'TickDir','out','Xscale','log','TickLength',3*get(gca,'TickLength'));
        
end

set(gca,'FontSize',14,'TickDir','out');
% set(gca,'TickDir','out','Yscale','log','TickLength',3*get(gca,'TickLength'));
box off;

set(gcf,'Color','w');

switch lower(options.Method)
    case 'ir'
        xstring = 'Func. evals. / Dim';
        ystring = 'Median IR';
    case 'fs'
        xstring = 'Function evaluations / D';
        ystring = 'Fraction solved';
    case 'ert'
        xstring = 'FEvals / Dim';
        ystring = 'Fraction solved';        
    case 'fst'
        % xstring = 'Error tolerance from maximum LL';
        xstring = ['Error tolerance ' char(949)];
        if isfield(options,'FunEvalsPerD') && ~isempty(options.FunEvalsPerD)
            ystring = ['Fraction solved at ' num2str(options.FunEvalsPerD) '×D func. evals.'];
        else
            ystring = 'Fraction solved';            
        end
end
xlabel(xstring,'FontSize',16);
ylabel(ystring,'FontSize',16);
if isfield(options,'VerticalThreshold') && ~isempty(options.VerticalThreshold)
    plot(options.VerticalThreshold*[1,1],[0,1],'k--','LineWidth',1);
end

% Add legend
[~,ord] = sort(perf,'descend');
hlines = [];
for iField = ord
    temp = fieldname{iField};
    idx = [find(temp == '_'), numel(temp)+1, numel(temp)+1];
    first = temp(idx(1)+1:idx(2)-1);
    %second = temp(idx(2)+1:idx(3)-1);
    second = temp(idx(2)+1:end);
    second(second == '_') = '-';
    if ~strcmpi(second,'base')
        legendlist{iField} = [first, ' (', second, ')'];
    else
        legendlist{iField} = first;
    end
    h = plot(min(xlims)*[1 1]/100,min(ylims)*[1 1]/100,[style(iField).linestyle,style(iField).marker],...
        'MarkerSize',markersize,'LineWidth',style(iField).linewidth,'Color',style(iField).color); hold on;    
    % h = shadedErrorBar(min(xlims)*[1 1],min(ylims)*[1 1],[0 0],{linstyle{iField},'LineWidth',2},1); h = h.mainline; hold on;
    hlines(end+1) = h;
end
hl = legend(hlines,legendlist{ord});
set(hl,'Box','on','Location','NorthWest','FontSize',16,'EdgeColor','none');

if reverseX_flag
    set(gca,'Xdir','reverse'); 
    set(hl,'Location','NorthEast');
end

set(gcf,'Position', [1 41 1920 958]);