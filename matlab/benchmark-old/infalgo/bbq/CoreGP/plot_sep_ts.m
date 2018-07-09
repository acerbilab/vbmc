function [sd,obs,mn,reals]=plot_sep_ts(XsFull,YsFull,XStars,YMean,YSD,XsReal,YsReal,xlab,ylab,titl,ax)

NDims=length(unique(XsFull(:,1)));
scrsz = get(0,'ScreenSize');

for gd=1:NDims
    
    figure('Position',[1 1 0.315*scrsz(3) 0.26*scrsz(4)])
    hold on
    box on
    set(gca,'FontName','Times','FontSize',20);
    
    NumStar=find(XStars(:,1)==gd);
    sd=shaded_sd(XStars(NumStar,2),YMean(NumStar),YSD(NumStar),[-0.15 -0.15 -0.15]);
    mn=plot(XStars(NumStar,2),YMean(NumStar),'b');
    Nums=find(XsFull(:,1)==gd);
    obs=plot(XsFull(Nums,2),YsFull(Nums),'dk','MarkerSize',8,'MarkerFaceColor','k');
    NumsReal=find(XsReal(:,1)==gd);
    reals=plot(XsReal(NumsReal,2),YsReal(NumsReal),'k');

    if length(xlab)>5
        Rot=90;
    else
        Rot=0;
    end
    
    if nargin>7
        ylabel(ylab,'FontName','Times','FontSize',22,'Rotation',Rot, 'Interpreter','latex');
    if nargin>8
        xlabel(xlab,'FontName','Times','FontSize',22);
    if nargin>9
       title(titl{gd},'FontName','Times','FontSize',26);
    if nargin>10
        axis(ax);
    end
    end
    end
    end

end