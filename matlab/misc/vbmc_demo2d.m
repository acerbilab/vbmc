function vbmc_demo2d(stats,fun)
%VBMC_DEMO2D Demo plot of VBMC at work (only for 2D problems).


% rng(0); [vp,elbo,elbo_sd,exitflag,output,optimState,stats] = vbmc(@(x) rosenbrock_test(x),[-1 -1],-Inf,Inf,-3,3,struct('Plot','on'));

if nargin < 2 || isempty(fun); fun = @rosenbrock_test; end

LB = [-3,-2];
UB = [3,6];
tolx = 1e-3;
Nx = 128;
Npanels = 8;

x1 = linspace(LB(1)+tolx,UB(1)-tolx,Nx);
x2 = linspace(LB(2)+tolx,UB(2)-tolx,Nx);
dx1 = x1(2)-x1(1);
dx2 = x2(2)-x2(1);

idx = ones(1,Npanels-2);
idx(2) = find(stats.warmup == 1,1,'last');
tmp = floor(linspace(idx(2),numel(stats.vp),Npanels-3));
idx(3:Npanels-2) = tmp(2:end);

grid = [reshape(1:Npanels-2,[(Npanels-2)/2,2])',[Npanels;Npanels-1]];
h = plotify(grid,'gutter',[0.05 0.15],'margins',[.05 .02 .075 .05]);

for iPlot = 1:Npanels
    axes(h(iPlot));
    
    %[X1,X2] = meshgrid(x1,x2);
    %tmp = cat(2,X2',X1');
    %xx = reshape(tmp,[],2);
    xx = combvec(x1,x2)';
    
    if iPlot <= numel(idx); vpflag = true; else vpflag = false; end
    
    elboflag = false;
    if vpflag
        vp = stats.vp(idx(iPlot));
        yy = vbmc_pdf(xx,vp);
        titlestr = ['Iteration ' num2str(stats.iter(idx(iPlot)))];
        if iPlot == 2; titlestr = [titlestr ' (end of warm-up)']; end
    elseif iPlot == Npanels-1
        lnyy = zeros(size(xx,1),1);
        for ii = 1:size(xx,1)
            lnyy(ii) = fun(xx(ii,:));
        end
        yy = exp(lnyy);
        Z = sum(yy(:))*dx1*dx2;
        yy = yy/Z;
        titlestr = ['True posterior'];
    else
        elboflag = true;
    end
    
    if elboflag
        iter = stats.iter;
        elbo = stats.elbo;
        elbosd = stats.elboSD;
        beta = 1.96;
        patch([iter,fliplr(iter)],[elbo + beta*elbosd, fliplr(elbo - beta*elbosd)],[1 0.8 0.8],'LineStyle','none'); hold on;
        hl(1) = plot(iter,elbo,'r','LineWidth',1); hold on;
        hl(2) = plot([iter(1),iter(end)],log(Z)*[1 1],'k','LineWidth',1);
        titlestr = 'Model evidence';
        xlim([0.9, stats.iter(end)+0.1]);
        ylims = [floor(min(elbo)-0.1),ceil(max(elbo)+0.1)];
        ylim(ylims);
        xticks(idx);
        yticks([ylims(1),round(log(Z),2),ylims(2)])
        xlabel('Iterations');
        
        hll = legend(hl,'ELBO','lnZ');
        set(hll,'Location','SouthEast','Box','off');
        
    else
        s = contour(x1,x2,reshape(yy',[Nx,Nx])');

        if vpflag
            % Plot component centers
            mu = warpvars(vp.mu','inv',vp.trinfo);
            hold on;
            plot(mu(:,1),mu(:,2),'xr','LineStyle','none');

            % Plot data
            X = warpvars(stats.gp(idx(iPlot)).X,'inv',vp.trinfo);
            plot(X(:,1),X(:,2),'.k','LineStyle','none');
        end

        % s.EdgeColor = 'None';
        view([0 90]);
        xlabel('x_1');
        ylabel('x_2');
        set(gca,'XTickLabel',[],'YTickLabel',[]);
        
        xlim([LB(1),UB(1)]);
        ylim([LB(2),UB(2)]);
    end    
    
    title(titlestr);
    set(gca,'TickDir','out');
end

set(gcf,'Color','w');

pos = [20,20,900,450];
set(gcf,'Position',pos);
set(gcf,'Units','inches'); pos = get(gcf,'Position');
set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
drawnow;


end