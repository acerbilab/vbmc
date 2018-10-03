function vbmc_plot2d(vp,LB,UB,gp,plotflag)
%VBMC_PLOT2D 2-D Plot of variational/target posterior.

if nargin < 4; gp = []; end
if nargin < 5 || isempty(plotflag); plotflag = true; end

tolx = 1e-3;
Nx = 128;

x1 = linspace(LB(1)+tolx,UB(1)-tolx,Nx);
x2 = linspace(LB(2)+tolx,UB(2)-tolx,Nx);
dx1 = x1(2)-x1(1);
dx2 = x2(2)-x2(1);

xx = combvec(x1,x2)';

if isa(vp,'function_handle'); fun = vp; vpflag = false; else; vpflag = true; end

if vpflag
    yy = vbmc_pdf(vp,xx);
else
    lnyy = zeros(size(xx,1),1);
    for ii = 1:size(xx,1)
        lnyy(ii) = fun(xx(ii,:));
    end
    yy = exp(lnyy);
    Z = sum(yy(:))*dx1*dx2;
    yy = yy/Z;
end

s = contour(x1,x2,reshape(yy',[Nx,Nx])');

if vpflag
    % Plot component centers
    if plotflag
        mu = warpvars(vp.mu','inv',vp.trinfo);
        hold on;
        plot(mu(:,1),mu(:,2),'xr','LineStyle','none');
    end

    % Plot data
    if ~isempty(gp)
        X = warpvars(gp.X,'inv',vp.trinfo);
        plot(X(:,1),X(:,2),'.k','LineStyle','none');
    end
end

% s.EdgeColor = 'None';
view([0 90]);
xlabel('x_1');
ylabel('x_2');
set(gca,'XTickLabel',[],'YTickLabel',[]);

xlim([LB(1),UB(1)]);
ylim([LB(2),UB(2)]);
set(gca,'TickLength',get(gca,'TickLength')*2);

set(gca,'TickDir','out');
set(gcf,'Color','w');

end