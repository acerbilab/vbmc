function [sd,obs,mn] = laplot_nice(name,XsFull,ZsFull,XStars,YMean,YSD,xlab,ylab,tit,Colour,plotsize)

if size(XsFull,2)==1
    XsFull = allcombs({1,XsFull});
end
NDims=length(unique(XsFull(:,1)));
if size(XStars,2)==1
    XStars = allcombs({1,XStars});
end



scrsz = get(0,'ScreenSize');
set(0,'defaulttextinterpreter','none')

isColour = nargin>9;

if nargin<11
    plotsize = 'wide';
end

switch plotsize
    case 'small'
        width = 7;
    case 'wide'
        width = 15;
    case 'really_wide'
        width = 15;
end

for gd=1:NDims
    
    switch plotsize
        case 'small'
            figure('Position',[1 1 0.14*scrsz(3) 0.25*scrsz(4)])%0.262*scrsz(4)])
        case 'wide'
            figure('Position',[1 1 0.3*scrsz(3) 0.3*scrsz(4)])
            %figure('Position',[1 1 0.945*scrsz(3) 0.5*scrsz(4)])
            %axis([0 11 0 250])
        case 'really_wide'
            figure('Position',[1 1 0.5*scrsz(3) 0.3*scrsz(4)])
            %figure('Position',[1 1 0.945*scrsz(3) 0.5*scrsz(4)])
            %axis([0 11 0 250])
    end
    hold on
    box on
    set(gca,'FontName','Times','FontSize',13);
    
    
    Nums=find(XsFull(:,1)==gd);
    
    if ~isempty(XStars)
        NumStar=find(XStars(:,1)==gd);
        if ~isColour
            sd = shaded_sd(XStars(NumStar,2),YMean(NumStar),YSD(NumStar),[0 0 0]);
            mn = plot(XStars(NumStar,2),YMean(NumStar),'b');
        elseif strcmp(Colour,'Grey')
            sd = shaded_sd(XStars(NumStar,2),YMean(NumStar),YSD(NumStar),[0 0 0]);
            mn = plot(XStars(NumStar,2),YMean(NumStar),'Color',[0.3 0.3 0.3]);            
        else
            sd = shaded_sd(XStars(NumStar,2),YMean(NumStar),YSD(NumStar),Colour);
            mn = plot(XStars(NumStar,2),YMean(NumStar),'Color',Colour);
        end
    end
    obs = plot(XsFull(Nums,2),ZsFull(Nums),'+k','LineWidth',2,'MarkerSize',12,'MarkerFaceColor','k'); %'MarkerSize',8
    Nums2=find(XsFull(:,1)==(mod(gd,2)+1));
    obs2 = plot(XsFull(Nums2,2),ZsFull(Nums2),':k','MarkerSize',12, 'LineWidth',2);
    
    if length(ylab)>3
        Rotation = 90;
    else
        Rotation = 0;
    end

    ylabel(ylab,'Rotation',Rotation);
    xlabel(xlab);
    title(tit);
    %title([tit,' ',num2str(gd)],'FontName','Times','FontSize',24);
end



laprint(gcf,name,'width',width,'factor',1,'scalefonts','off',...
    'keepfontprops','off','asonscreen','off','keepticklabels','off',...
    'mathticklabels','off','head','off','caption','caption',...
    'figcopy', 'on')