function [sd,obs,mn] = plot_nice(XsFull,ZsFull,XStars,YMean,YSD,xlab,ylab,tit,Colour,plotsize)

if size(XsFull,2)==1
    XsFull = allcombs({1,XsFull});
end
NDims=length(unique(XsFull(:,1)));
if size(XStars,2)==1
    XStars = allcombs({1,XStars});
end


scrsz = get(0,'ScreenSize');

isColour = nargin>8;
if ~isColour
    Colour = [1 0 0];
end

if nargin<10
    plotsize = 'small';
end

for gd=1:NDims
    
    %figure
    switch plotsize
    case 'small'
        width = 7;
        height = 5;
        MarkerSize = 3;
    case 'wide'
        width = 15;
        height = 5;
        MarkerSize = 3;
    case 'NIPS'
        width = 14;
        height = 4;
        MarkerSize = 2;
    end
    set(gcf,'Position',[1 1 width/(2*37.7)*scrsz(3) height/30*scrsz(4)])
    
%     switch plotsize
%         case 'small'
%             figure('Position',[1 1 0.2*scrsz(3) 0.4*scrsz(4)])%0.262*scrsz(4)])
%         case 'wide'
%             figure('Position',[1 1 0.9*scrsz(3) 0.4*scrsz(4)])
%             %figure('Position',[1 1 0.945*scrsz(3) 0.5*scrsz(4)])
%             %axis([0 11 0 250])
%         case 'smallwide'
%             figure('Position',[1 1 0.4725*scrsz(3) 0.3*scrsz(4)])
%             %axis([0 11 0 250])
%     end
    hold on
    box on
    set(gca,'FontName','Times','FontSize',13);
    
    
    Nums=find(XsFull(:,1)==gd);
    
    if ~isempty(XStars)
        NumStar=find(XStars(:,1)==gd);
        if strcmp(Colour,'Grey')
            sd = shaded_sd(XStars(NumStar,2),YMean(NumStar),YSD(NumStar),[0 0 0]);
            mn = plot(XStars(NumStar,2),YMean(NumStar),'Color',[0.3 0.3 0.3]);            
        else
            sd = shaded_sd(XStars(NumStar,2),YMean(NumStar),YSD(NumStar),Colour);
            mn = plot(XStars(NumStar,2),YMean(NumStar),'Color',Colour);
        end
    end
    obs = plot(XsFull(Nums,2),ZsFull(Nums),'+k','MarkerSize',MarkerSize,'MarkerFaceColor','k'); %'MarkerSize',8
    Nums2=find(XsFull(:,1)==(mod(gd,2)+1));
    obs2 = plot(XsFull(Nums2,2),ZsFull(Nums2),':k','MarkerSize',MarkerSize, 'LineWidth',1);
    
    if length(ylab)>3
        Rotation = 90;
    else
        Rotation = 0;
    end

    ylabel(ylab,'FontName','Times','FontSize',13,'Rotation',Rotation);
    xlabel(xlab,'FontName','Times','FontSize',13);
    title(tit,'FontName','Times','FontSize',13);
    %title([tit,' ',num2str(gd)],'FontName','Times','FontSize',24);
end