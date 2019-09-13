function vbmc_iterplot(vp,gp,optimState,stats,elbo)
%VBMC_ITERPLOT Plot current iteration of the VBMC algorithm.

D = vp.D;
iter = optimState.iter;
fontsize = 14;

if D == 1
    hold off;
    gplite_plot(gp);
    hold on;
    xlims = xlim;
    xx = linspace(xlims(1),xlims(2),1e3)';
    yy = vbmc_pdf(vp,xx,false,true);
    hold on;
    plot(xx,yy+elbo,':');
    drawnow;

else
    if ~isempty(vp)
        Xrnd = vbmc_rnd(vp,1e5,1,1);
    else
        Xrnd = gp.X;
    end
    X_train = gp.X;
    
    if iter == 1
        idx_new = true(size(X_train,1),1);
    else
        X_trainold = stats.gp(iter-1).X;
        idx_new = false(size(X_train,1),1);
        [~,idx_diff] = setdiff(X_train,X_trainold,'rows');
        idx_new(idx_diff) = true;
    end
    idx_old = ~idx_new;

    if ~isempty(vp.trinfo); X_train = warpvars_vbmc(X_train,'inv',vp.trinfo); end
    
    Pdelta = optimState.PUB_orig - optimState.PLB_orig;
    X_min = min(X_train,[],1) - Pdelta*0.1;
    X_max = max(X_train,[],1) + Pdelta*0.1;    
    bounds = [max(min(optimState.PLB_orig,X_min),optimState.LB_orig); ...
        min(max(optimState.PUB_orig,X_max),optimState.UB_orig)];    
    
    try
        for i = 1:D; names{i} = ['x_{' num2str(i) '}']; end
        [~,ax] = cornerplot(Xrnd,names,[],bounds);
        for i = 1:D-1
            for j = i+1:D
                axes(ax(j,i));  hold on;
                if any(idx_old)
                    scatter(X_train(idx_old,i),X_train(idx_old,j),'ok');                            
                end
                if any(idx_new)
                    scatter(X_train(idx_new,i),X_train(idx_new,j),'or','MarkerFaceColor','r');                            
                end
            end
        end
        
        h = axes(gcf,'Position',[0 0 1 1]);
        set(h,'Color','none','box','off','XTick',[],'YTick',[],'Units','normalized','Xcolor','none','Ycolor','none');
        text(0.9,0.9,['VBMC (iteration ' num2str(iter) ')'],'FontSize',fontsize,'HorizontalAlignment','right');
        
        drawnow;
    catch
        % pause
    end
end

end