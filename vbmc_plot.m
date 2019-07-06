function vbmc_plot(vp_array,stats)

if nargin < 2; stats = []; end

Nsamples = 1e5;

if ~iscell(vp_array)
    temp{1} = vp_array;
    vp_array = temp;
end

if numel(vp_array) == 1 && vbmc_isavp(vp_array{1})        
    X = vbmc_rnd(vp_array{1},Nsamples);
    for d = 1:size(X,2); names{d} = ['x_{' num2str(d) '}']; end
    cornerplot(X,names);
else
    Nbins = 40;
    Nvps = numel(vp_array);
    D = vp_array{1}.D;
    mm = zeros(Nvps,D);
    cmap = colormap;
    cmap = cmap(mod((1:27:(1+27*64))-1,64)+1,:);
    
    plotmat = [1 1; 1 2; 1 3; 2 2; 2 3; 2 3; 2 4; 2 4; 3 3; 3 4; 3 4; 3 4; 3 5; 3 5; 3 5; 4 4; 4 5; 4 5; 4 5];    
    nrows = plotmat(D,1);
    ncols = plotmat(D,2);
    
    for i = 1:Nvps
        if ~isempty(stats) && stats.idx_best == i; best_flag = true; else; best_flag = false; end
        ltext{i} = ['vp #' num2str(i)];
        if best_flag; ltext{i} = [ltext{i} ' (best)']; end
        
        X = vbmc_rnd(vp_array{i},Nsamples);
        mm(i,:) = median(X);
        
        for d = 1:D
            subplot(nrows,ncols,d);            
            if best_flag; lw = 3; else; lw = 1; end
            hst(i)=histogram(X(:,d),Nbins,'Normalization','probability','Displaystyle','stairs','LineWidth',lw,'EdgeColor',cmap(i,:));
            hold on;
        end
    end
    
    for i = 1:Nvps
        if ~isempty(stats) && stats.idx_best == i; best_flag = true; else; best_flag = false; end
        for d = 1:D
            subplot(nrows,ncols,d);            
            if best_flag; lw = 3; else; lw = 1; end
            hln(i)=plot(mm(i,d)*[1 1],ylim,'-','LineWidth',lw,'Color',cmap(i,:));
            hold on;
        end
    end
    
    
    for d = 1:D
        subplot(nrows,ncols,d);            
        
        xlabel(['x_{' num2str(d) '}']);
        set(gca,'TickDir','out');
        box off;
        
        if d == D
            hleg = legend(hln,ltext{:});
            set(hleg,'box','off','location','best');
        end
        
    end
    set(gcf,'Color','w');
    
end




end