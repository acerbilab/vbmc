function benchdata = infbench_plot(varargin)
%INFBENCH_PLOT Display inference benchmark results.
%
%   INFBENCH_PLOT(PROBSET,PROB,SUBPROB,NOISE,ALGO,ALGOSET,ORDER) 
%   factorially plots optimization benchmark results for the following 
%   factors: problem set PROBSET, problem(s) PROB, subproblem(s) SUBPROB, 
%   noise level NOISE, algorithm(s) ALGO, algorithm setting(s) ALGOSET. 
%   PROBSET, PROB, SUBPROB, NOISE, ALGO, and ALGOSET can be strings or cell 
%   arrays of strings. ORDER is a cell array of string such that: ORDER{1} 
%   is the factor expanded across rows, ORDER{2} is the factor expanded 
%   across columns, and ORDER{3} (if present) is the factor expanded across
%   figures. Unassigned factors are plotted in the same panel. The factors
%   are 'probset', 'prob', 'subprob', 'noise', 'algo', and 'algoset'.
%
%   BENCHMARK_PLOT(PROBSET,PROB,SUBPROB,NOISE,ALGO,ALGOSET,ORDER,BESTOUTOF) 
%   plots an estimate of the 'minimum out of BESTOUTOF runs'. By default
%   BESTOUTOF is 1.
%
%   Example
%      A typical usage of INFBENCH_PLOT:
%         infbench_plot('vbmc18',{'lumpy','cigar'},...
%         {'2D','4D','8D'},[],{'vbmc@acqvar','wsabi'},'base',{'prob','subprob'})
%      plots 'lumpy' benchmark on the first row and 'cigar' on the second 
%      row, each column is a different dimension for D=2,4,8, and each panel 
%      compares 'vbmc' (variance-based acquisition fcn) and 'wsabi'.
%
%   See also INFBENCH_RUN.

% Luigi Acerbi 2018

% Base options
defopts.BestOutOf = 1;
defopts.ErrorBar = [];
defopts.NumZero = 1e-8;
defopts.Method = 'IR';          % Immediate regret (IR) or fraction solved (FS)
defopts.SolveThreshold = 1e-6;
defopts.FileName = ['.' filesep 'infbenchdata.mat'];
defopts.Nsamp = 5e3;            % Samples for ERT computation
defopts.TwoRows = 0;
defopts.EnhanceLine = 0;        % Enhance one plotted line
defopts.FunEvalsPerD = 500;
defopts.PlotType = 'nlZ';
defopts.DisplayLegend = true;
defopts.Quantiles = [0.25,0.75];    % Confidence intervals quantiles
defopts.PlotAll = false;        % Plot all lines
defopts.BootStrap = 1e3;        % # samples for bootstrap
% defopts.BaseSpeedTest = 8.2496; % Laptop speed
defopts.BaseSpeedTest = 13.7660; % Laptop speed

% Plotting options
defopts.YlimMax = 1e5;
defopts.AbsolutePlot = 0;
defopts.DisplayFval = 0;

StatMismatch = 0;

defaults = infbench_defaults('options');
linstyle = defaults.LineStyle;
lincol = defaults.LineColor;

if isstruct(varargin{end}); options = varargin{end}; else; options = []; end

% Assign default values to OPTIONS struct
for f = fieldnames(defopts)'
    if ~isfield(options,f{:}) || isempty(options.(f{:}))
        options.(f{:}) = defopts.(f{:});
    end
end

BestOutOf = options.BestOutOf;
NumZero = options.NumZero;

labels = {'probset','prob','subprob','noise','algo','algoset'};
for i = 1:length(labels); varargin{i} = cellify(varargin{i}); end

% linstyle = {'k-','b-','r-','g-','m:','r:','c:','g:','k-.','r-.'};

order = varargin{7};

% Find out what to plot on rows, columns and figures
dimrows = order{1};
if ischar(dimrows); dimrows = find(strcmp(labels,dimrows),1); end
nrows = numel(varargin{dimrows});

dimcols = order{2};
if ischar(dimcols); dimcols = find(strcmp(labels,dimcols),1); end
ncols = numel(varargin{dimcols});

if length(order) > 2
    dimfig = order{3};
    if ischar(dimfig); dimfig = find(strcmp(labels,dimfig),1); end
    nfigs = numel(varargin{dimfig});
else
    nfigs = 1;
    dimfig = [];
end

for i = 1:numel(labels); benchlist{i} = []; end

dimlayers = [];

% Load data from file
data = [];
try
    if exist(options.FileName,'file') == 2
        load(options.FileName,'data');
        fprintf('Loaded stored data from file.\n');
        LoadedData_flag = true;
    end
catch
    % Did not work
end
if isempty(data)
    fprintf('Stored data file does not exist. Loading raw data...\n');
    LoadedData_flag = false;
end

% Loop over figures
for iFig = 1:nfigs
    if nfigs > 1; benchlist{dimfig} = varargin{dimfig}{iFig}; end

    % Loop over rows
    for iRow = 1:nrows
        benchlist{dimrows} = varargin{dimrows}{iRow};
        if isempty(benchlist{dimrows}); continue; end
        
        % Loop over columns
        for iCol = 1:ncols 
            benchlist{dimcols} = varargin{dimcols}{iCol};
            if isempty(benchlist{dimcols}); continue; end
            
            if nrows == 1 && options.TwoRows
                index = iCol + (iCol > ceil(ncols/2));
                subplot(2,ceil(ncols/2)+1,index);
            else
                subplot(nrows,ncols+1,(iRow-1)*(ncols+1) + iCol);
            end
            cla(gca,'reset');
            
            % Find data dimension to be plotted in each panel
            for i = 1:length(labels)
                if isempty(benchlist{i})
                    benchlist{i} = varargin{i}; 
                    if numel(benchlist{i}) > 1 && isempty(dimlayers)
                        dimlayers = i;
                    else
                        benchlist{i} = varargin{i}{1};
                    end
                end
            end
            if isempty(dimlayers)
                dims = setdiff(1:length(labels),[dimfig dimrows dimcols]);
                dimlayers = dims(1); 
            end
                                    
            IsMinKnown = true;
            MinFvalNew = Inf;
            
            % Initialize summary statistics (load from file if present)
            benchdata = loadSummaryStats( ...
                options.FileName,benchlist,varargin{dimlayers},dimlayers);
                                                
            if IsMinKnown; MinPlot = NumZero; end            
                        
            % Loop over layers within panel
            for iLayer = 1:numel(varargin{dimlayers})
                
                % Collect history structs from data files
                benchlist{dimlayers} = varargin{dimlayers}{iLayer};
                fieldname = [benchlist{dimrows} '_' benchlist{dimcols} '_' benchlist{dimlayers}];
                fieldname(fieldname == '@') = '_';
                fieldname(fieldname == '-') = '_';
                flags = 0;
                if LoadedData_flag && isfield(data,fieldname)
                    history = data.(fieldname).history;
                    algo = data.(fieldname).algo;
                    algoset = data.(fieldname).algoset;
                    if numel(algoset) > 9 && strcmp(algoset(end-8:end), '_overhead')
                        flags(1) = 1;                        
                    end
                else
                    fprintf('Collecting files: %s@%s@%s\n', benchlist{dimrows}, benchlist{dimcols}, benchlist{dimlayers});
                    [history,algo,algoset,flags,Nfiles] = collectHistoryFiles(benchlist);
                    data.(fieldname).history = history;
                    data.(fieldname).algo = algo;
                    data.(fieldname).algoset = algoset;
                end
                algoset(algoset == '-') = '_';
                                
                if flags(1)
                    overhead_flag = true;
                else
                    overhead_flag = false;
                end
                
                if isempty(history); continue; end
        
                
                x = []; lnZs = []; gsKLs = []; D = []; FunCallsPerIter = [];
                AverageOverhead = zeros(1,numel(history));
                FractionOverhead = [];
                Errs = []; Zscores = [];
                TotalElapsedTime = 0;
                TotalFunctionTime = 0;
                TotalTrials = 0;
                                
                % Loop over histories
                for i = 1:numel(history)
                    
                    D = [D; history{i}.D];  % Number of variables
                    
                    x = [x; history{1}.SaveTicks];

                    % Get valid time ticks
                    idx_valid = history{i}.SaveTicks <= history{i}.TotalMaxFunEvals;                        
                    lZs_new = history{i}.Output.lnZs(idx_valid);    lZs_new = lZs_new(:)';                   
                    lZs_var_new = history{i}.Output.lnZs_var(idx_valid);    lZs_var_new = lZs_var_new(:)';                 
                    gsKLs_new = history{i}.Output.gsKL(idx_valid);                        

                    lnZ_true = history{i}.lnZpost_true;                
                    
                    lnZs = [lnZs; lZs_new];
                    gsKLs = [gsKLs; gsKLs_new];
                    
                    Errs = [Errs; abs(lZs_new - lnZ_true)];
                    Zscores = [Zscores; (lZs_new - lnZ_true)./ sqrt(lZs_var_new)];
                    
                    if ~isempty(history{i})
                        
                        if isfield(history{i},'FunCallsPerIter')
                            FunCallsPerIter{i} = history{i}.FunCallsPerIter;
                        else
                            FunCallsPerIter{i} = NaN; 
                        end
                        
                        % Normalize to laptop speed
                        speedfactor = options.BaseSpeedTest/history{i}.speedtest;                        
                        
                        if isfield(history{i},'SaveTicks')
                            % Take only until it resets (one run)
                            last = find(isfinite(history{i}.ElapsedTime),1,'last');
                            last_first = find(diff(history{i}.ElapsedTime(1:last)) < 0,1);
                            if ~isempty(last_first); last = last_first; end
                            try
                                AverageOverhead(i) = speedfactor * ...
                                    (history{i}.ElapsedTime(last) - sum(history{i}.FuncTime(1:last)))/history{1}.SaveTicks(last);
                            catch
                                pause
                            end
                        end                        
                        
                        TotalElapsedTime = TotalElapsedTime + history{i}.ElapsedTime(last)*speedfactor;
                        TotalFunctionTime = TotalFunctionTime + sum(history{i}.FuncTime(1:last))*speedfactor;
                        TotalTrials = TotalTrials + history{i}.SaveTicks(last);
                        FractionOverhead = [FractionOverhead, (history{i}.ElapsedTime(last)/sum(history{i}.FuncTime(1:last))-1)];                        
                    end
                    
                end
                                
                % Save summary statistics
                if isempty(benchlist{4}); noise = [];
                else noise = ['_' benchlist{4} 'noise']; end                
                field1 = ['f1_' benchlist{1} '_' benchlist{2}];
                field2 = ['f2_' upper(benchlist{3}) noise];
                field3 = ['f3_' algo '_' algoset];
                
                itersPerRun = cellfun(@length,FunCallsPerIter);
                fprintf('Loaded files: %d. History runs collected: %d.\n', Nfiles, numel(history));
                display(['Average # of algorithm starts per run: ' num2str(mean(itersPerRun)) ' ± ' num2str(std(itersPerRun)) '.']);                
                display(['Average overhead per function call: ' num2str(mean(AverageOverhead),'%.3f') ' ± ' num2str(std(AverageOverhead),'%.3f') '.']);
                
%                 if any(AverageOverhead < 0)
%                     pause
%                 end
                
                if ~isempty(history)

                    if overhead_flag
                        for iRun = 1:size(x,1)
                            y(iRun,:) = interp1(x(iRun,:),y(iRun,:),x(iRun,:)/(1+FractionOverhead(iRun)));
                        end
                        fprintf('Fraction overhead: %.3f +/- %.3f.\n', mean(FractionOverhead), std(FractionOverhead));
                    end
                    if ~options.AbsolutePlot
                        lnZs = Errs;
                    end
                    
                    
                    
                    switch lower(options.PlotType)
                        case 'lnz'
                            [xx,yy,yyerr_up,yyerr_down] = plotIterations(x,lnZs,iLayer,varargin{dimlayers},options);
                        case 'gskl'
                            [xx,yy,yyerr_up,yyerr_down] = plotIterations(x,gsKLs,iLayer,varargin{dimlayers},options);
                    end
                    
                    % Save summary information
                    benchdatanew.(field1).(field2).(field3).xx = xx;
                    benchdatanew.(field1).(field2).(field3).yy = yy;
                    benchdatanew.(field1).(field2).(field3).yerr_up = yyerr_up;
                    benchdatanew.(field1).(field2).(field3).yerr_down = yyerr_down;
                    benchdatanew.(field1).(field2).(field3).MaxFunEvals = ...
                        history{1}.TotalMaxFunEvals;
                    
                    benchdatanew.(field1).(field2).(field3).AverageAlgTime = ...
                        (TotalElapsedTime - TotalFunctionTime)/TotalTrials;
                    benchdatanew.(field1).(field2).(field3).AverageFunTime = ...
                        TotalFunctionTime/TotalTrials;
                    benchdatanew.(field1).(field2).(field3).FractionOverhead = mean(FractionOverhead);
%                     benchdatanew.(field1).(field2).(field3).Zscores = Zscores(:);
%                     benchdatanew.(field1).(field2).(field3).Errs = Errs(:);
                end
                
            end
            
            [xlims,ylims] = panelIterations(iRow,iCol,nrows,ncols,dimrows,dimcols,xx,lnZ_true,benchlist,IsMinKnown,options);
        end
    end
    
    %% Add legend
    if options.DisplayLegend
        if nrows == 1 && options.TwoRows
            subplot(nrows,ceil(ncols/2)+1,ceil(ncols/2)+1);
        else
            subplot(nrows,ncols+1,ncols+1);
        end
    
        cla(gca,'reset');
        for iLayer = 1:length(varargin{dimlayers})
            temp = varargin{dimlayers}{iLayer};
            index = find(temp == '@',1);
            if isempty(index)
                legendlist{iLayer} = temp;
            else
                first = temp(1:index-1);
                second = temp(index+1:end);
                if ~strcmpi(second,'base')
                    legendlist{iLayer} = [first, ' (', second, ')'];
                else
                    legendlist{iLayer} = first;                
                end
            end
            
            style = infbench_defaults('style',[],[],temp);
            linstyle = style.linestyle;
            lincol = style.color;
            linewidth = style.linewidth;
            linemarker = []; %style.marker;
            if ~isempty(style.name); legendlist{iLayer} = style.name; end
            
            enhance = enhanceline(length(varargin{dimlayers}),options);
            if any(iLayer == enhance); lw = 2*linewidth; else; lw = linewidth; end
            % h = shadedErrorBar(min(xlims)*[1 1],min(ylims)*[1 1],[0 0],{linstyle{iLayer},'Color',lincol(iLayer,:),'LineWidth',lw},1); hold on;
            h = plot(min(xlim)*[1 1],min(ylim)*[1 1],[linstyle,linemarker],'Color',lincol,'LineWidth',lw); hold on;
            hlines(iLayer) = h;
        end
        hl = legend(hlines,legendlist{:});
        set(hl,'Box','off','Location','NorthWest','FontSize',14);
        axis off;
    end
    
    set(gcf,'Color','w');
end

% if StatMismatch == 1 || isempty(benchdata)
%     warning('Summary statistics are not up to date. Replot to have the correct graphs.');
% end

benchdata = benchdatanew;
benchdata.options = options;

% save(options.FileName,'benchdata','data');

%--------------------------------------------------------------------------
function [xx,yy,yyerr_up,yyerr_down] = plotIterations(x,y,iLayer,arglayer,options)
%PLOTITERATIONS Plot time series of IR or FS
        
    yyerr_up = []; yyerr_down = [];

    defaults = infbench_defaults('options');
    linstyle = defaults.LineStyle;
    lincol = defaults.LineColor;
    
    style = infbench_defaults('style',[],[],arglayer{iLayer});
    linstyle = style.linestyle;
    lincol = style.color;
    linewidth = style.linewidth;
    linemarker = []; %style.marker;
        
    xx = median(x,1);
    switch lower(options.Method)
        case 'ir'
            idx_bad = all(~isfinite(y),2);
            y(idx_bad,:) = [];
            
            if options.PlotAll
                qq = 0.05:0.05:0.95;
                for i = 1:numel(qq); yy(i,:) = quantile(y,qq(i)); end
            else
                yy = median(y,1);

                if options.BootStrap > 0
                    yy_boot = bootstrp(options.BootStrap,@median,y);
                    yyerr_up = quantile(yy_boot,0.95,1);
                    yyerr_down = quantile(yy_boot,0.05,1);
                else
                    yyerr_up = quantile(y,options.Quantiles(2),1);
                    yyerr_down = quantile(y,options.Quantiles(1),1);
                end

                idx = isfinite(yy);
                xx = xx(idx);
                yy = yy(idx);
                yyerr_up = yyerr_up(idx);
                yyerr_down = yyerr_down(idx);
            end
    end

    plotErrorBar = options.ErrorBar;
    if isempty(plotErrorBar)
        plotErrorBar = numel(arglayer) <= 3;
    end
        
    enhance = enhanceline(numel(arglayer),options);    

    if options.PlotAll
        lw = 0.5;
        for i = 1:size(yy,1)
            h = plot(xx,yy(i,:),[linstyle,linemarker],'Color', [lincol,0.5], 'LineWidth',lw); hold on;
        end
    else
        if any(iLayer == enhance); lw = linewidth*2; else; lw = linewidth; end    
        if plotErrorBar
            % h = shadedErrorBar(xx,yy,yyerr,{linstyle,'LineWidth',lw},1); hold on;

            xxerr = [xx, fliplr(xx)];
            yyerr = [yyerr_down, fliplr(yyerr_up)];
            fill(xxerr, yyerr,lincol,'FaceAlpha',0.5,'LineStyle','none'); hold on;

            h = plot(xx,yy,[linstyle,linemarker],'Color', lincol, 'LineWidth',lw); hold on;
            %plot(xx,yyerr_up,linstyle,'Color', lincol, 'LineWidth',1); hold on;
            %plot(xx,yyerr_down,linstyle,'Color', lincol, 'LineWidth',1); hold on;
        else
            h = plot(xx,yy,[linstyle,linemarker],'Color', lincol, 'LineWidth',lw); hold on;
        end
    end

%--------------------------------------------------------------------------
function [xlims,ylims] = panelIterations(iRow,iCol,nrows,ncols,dimrows,dimcols,xx,lnZ_true,benchlist,IsMinKnown,options)
%PANELITERATIONS Finalize panel for plotting iterations

    NumZero = options.NumZero;

    switch lower(options.PlotType)
        case 'lnz'
            ystring = 'Median LML error';
        case 'gskl'
            ystring = 'Median gsKL';
    end

    string = [];
    if options.DisplayFval; string = [' (lnZ_{true} = ' num2str(lnZ_true,'%.2f') ')']; 
    else string = []; end

    if iRow == 1; title([benchlist{dimcols} string]); end
    % if iCol == 1; ylabel(benchlist{dimrows}); end
    if iCol == 1; ylabel(ystring); end
    if iCol == 1
        textstr = benchlist{dimrows};
        textstr(textstr == '_') = ' ';
        text(-1/6*(dimcols+1),0.9,textstr,'Units','Normalized','Rotation',0,'FontWeight','bold','HorizontalAlignment','center');
    end
    xlims = [min(xx,[],2) max(xx,[],2)];
    if xlims(2) >= 400
        xtick = [200:200:1000];
    else
        xtick = [100:100:1000];
    end
    % xtick = [1e2,1e3,2e3,3e3,4e3,5e3,6e3,1e4,1e5];
    set(gca,'Xlim',xlims,'XTick',xtick)
    switch lower(options.Method)
        case 'ir'
            if IsMinKnown && 0
                ylims = [NumZero,options.YlimMax];
                if NumZero < 1e-5
                    ytick = [NumZero,1e-5,0.1,1,10,1e5,1e10];
                    yticklabel = {'0','10^{-5}','0.1','1','10','10^5','10^{10}'};
                else
                    ytick = [NumZero,0.1,1,10,1e3];                    
                    yticklabel = {'10^{-3}','0.1','1','10','10^3'};
                end
                liney = [1 1];
            else
                YlimMax = options.YlimMax;
                if options.AbsolutePlot
                    ylims = [lnZ_true-30,lnZ_true+30];
                    ytick = [];
                else
                    ylims = [NumZero,YlimMax];
                    %ytick = [0.001,0.01,0.1,1,10,100,1e3,1e4,1e5];
                    %yticklabel = {'0.001','0.01','0.1','1','10','10^2','10^3','10^4','10^5'};
                    ytick = [0.0001,0.01,1,100,1e4,1e6,1e8];
                    yticklabel = {'10^{-4}','10^{-2}','1','10^{2}','10^4','10^6','10^8'};
                end
                liney = lnZ_true*[1 1];                
            end
            if isempty(ytick)
                set(gca,'Ylim',ylims);                        
            else
                set(gca,'Ylim',ylims,'YTick',ytick,'YTickLabel',yticklabel);
            end
            
            if options.AbsolutePlot
                set(gca,'TickDir','out','TickLength',3*get(gca,'TickLength'));                
            else
                set(gca,'TickDir','out','Yscale','log','TickLength',3*get(gca,'TickLength'));                
                set(gca,'YMinorTick','off');
            end
            xstring = 'Function evaluations';            
    end
    if options.TwoRows
        plotXlabel = (iCol > ceil(ncols/2));
    else
        plotXlabel = (iRow == nrows);
    end
    if plotXlabel; xlabel(xstring); end
    set(gca,'FontSize',12);
    box off;
    % plot(xlims,liney,'k--','Linewidth',0.5);
    plot(xlims,[1 1],'k--','Linewidth',0.5);
    
%--------------------------------------------------------------------------
function [xx,yy,yyerr,MeanMinFval] = plotNoisy(y,MinBag,iLayer,arglayer,options)
%PLOTITERATIONS Plot time series of IR or FS

    NumZero = options.NumZero;
    BestOutOf = options.BestOutOf;
    
    defaults = benchmark_defaults('options');
    linstyle = defaults.LineStyle;
    lincol = defaults.LineColor;
        
    xx = iLayer;
    
    switch lower(options.Method)
        case 'ir'                        
            %yy = median(y,1);
            %yyerr = abs(bsxfun(@minus,[quantile(y,0.75,1);quantile(y,0.25,1)],yy));
        case 'fs'
            n = size(y,1);
            
            if all(MinBag.fsd == 0) && all(y(:,2) == 0)
                MeanMinFval = min(MinBag.fval);
                d = y(:,1) - MeanMinFval;                
            else            
                nn = 1000;
                Nsamples = numel(MinBag.fval);
                y = repmat(y, [ceil(Nsamples/n) 1]);
                y = y(randperm(Nsamples),:);
                fval = repmat(MinBag.fval,[1 nn]);
                fsd = repmat(MinBag.fsd,[1 nn]);
                f1 = bsxfun(@plus,y(:,1),bsxfun(@times,y(:,2),randn(size(y,1),nn)));
                fmin = min(fval + fsd.*randn(size(fsd)),[],1);
                MeanMinFval = nanmean(fmin);
                d = bsxfun(@minus, f1, fmin);
            end
            % target = nanmean(bsxfun(@lt, d(:), options.SolveThreshold(:)'),2);            
            target = bsxfun(@lt, d(:), options.SolveThreshold(:)');     
            yy = nanmean(target,1);
            yyerr = stderr(target,[],1);
    end

    plotErrorBar = options.ErrorBar;
    if isempty(plotErrorBar)
        plotErrorBar = numel(arglayer) <= 3;
    end
    if iLayer == numel(arglayer); lw = 4; else lw = 2; end
    
    if size(yy,2) == 1
        h = bar(xx,yy,'LineStyle','none','FaceColor',lincol(iLayer,:)); hold on;
    else
        xx = options.SolveThreshold;
        h = plot(xx,yy,'LineStyle','-','Color',lincol(iLayer,:)); hold on;
    end
    
    % h = errorbar(xx,yy,yyerr,linstyle{iLayer},'Color', lincol(iLayer,:),'LineWidth',lw); hold on;
    
    
    %if plotErrorBar && 0
    %    h = shadedErrorBar(xx,yy,yyerr,{linstyle{iLayer},'LineWidth',lw},1); hold on;
    %else
    %    h = plot(xx,yy,linstyle{iLayer},'Color', lincol(iLayer,:), 'LineWidth',lw); hold on;
    %end
    % MinFval = min(MinFval,min(y(:)));
    
%--------------------------------------------------------------------------
function [xlims,ylims] = panelNoisy(iRow,iCol,nrows,dimrows,dimcols,xx,MeanMinFval,benchlist,options)
%PANELITERATIONS Finalize panel for plotting iterations

    NumZero = options.NumZero;

    switch lower(options.Method)
        case 'ir'
            ystring = 'Median IR';
        case 'fs'
            ystring = 'Fraction solved';
    end

    if options.DisplayFval; string = [' (<f_{min}> = ' num2str(MeanMinFval,'%.2f') ')']; 
    else string = []; end

    if iRow == 1; title([benchlist{dimcols} string]); end
    % if iCol == 1; ylabel(benchlist{dimrows}); end
    if iCol == 1; ylabel(ystring); end
    if iRow == nrows; xlabel('Algorithms'); end
    if iCol == 1
        textstr = benchlist{dimrows};
        textstr(textstr == '_') = ' ';
        text(-1/6*(dimcols+1),0.9,textstr,'Units','Normalized','Rotation',0,'FontWeight','bold','HorizontalAlignment','center');
    end
    xlims = [0 10];
    xtick = [];
    set(gca,'Xlim',xlims,'XTick',xtick)
    switch lower(options.Method)
        case 'ir'
            YlimMax = options.YlimMax;
            if options.AbsolutePlot
                ylims = [MinFval,MinFval + YlimMax];
                ytick = [];
            else
                ylims = [NumZero,YlimMax];
                ytick = [0.001,0.01,0.1,1,10,100,1000];
                yticklabel = {'0.001','0.01','0.1','1','10','100','1000'};
            end
            liney = MinFval*[1 1];                
            if isempty(ytick)
                set(gca,'Ylim',ylims);                        
            else
                set(gca,'Ylim',ylims,'YTick',ytick,'YTickLabel',yticklabel);
            end
            set(gca,'TickDir','out','TickLength',3*get(gca,'TickLength'));

        case 'fs'
            ylims = [0,1];
            ytick = [0,0.5,1];
            yticklabel = {'0','0.5','1'};
            liney = [1 1];
            set(gca,'Ylim',ylims,'YTick',ytick,'YTickLabel',yticklabel);
            set(gca,'TickDir','out','TickLength',3*get(gca,'TickLength'));
    end
    set(gca,'FontSize',12);
    box off;
    % plot(xlims,liney,'k--','Linewidth',0.5);    

%--------------------------------------------------------------------------
function c = cellify(x)

    if ~iscell(x); c = {x}; else c = x; end 
