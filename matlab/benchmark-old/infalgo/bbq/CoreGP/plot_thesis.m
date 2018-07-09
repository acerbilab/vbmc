function plot_thesis(XsFull,ZsFull,XsData,ZsData,XStars,YMean,YSD,xlab,ylab,tit,axislimits,plotsize,name,othercmds,flag,MarkerLineWidth)
%['GP Assuming Simply',sprintf('\n'),'Noisy Observations'] inserts a
%linebreak into a text string

plot_full =  ~isempty(XsFull);

if size(XsData,2)==1
    XsData = allcombs({1,XsData});
end
if size(XsFull,2)==1 && plot_full
    XsFull = allcombs({1,XsFull});
end
NDims=length(unique(XsData(:,1)));
if size(XStars,2)==1
    XStars = allcombs({1,XStars});
end

if ~iscellstr(tit)
    stringy = tit;
    tit = cell(1,NDims);
    for i=1:NDims
        tit{i} = stringy;
    end
end


scrsz = get(0,'ScreenSize');

Colour = [1 0 0];

if nargin<14
    othercmds = '';
end

if nargin==15
    nolp = strcmpi(flag,'nolp');
else
    nolp = false;
end
enbiggen = 1+nolp;


if ischar(plotsize)
    switch plotsize
        case 'half'
            width = 7.5;
            height = 9;
        case 'full'
            width = 16;
            height = 10;
    end
    MarkerSize = 5;
else
    width = plotsize(1);
    height = plotsize(2);
    MarkerSize = plotsize(3);
end

if exist('MarkerLineWidth','var') == 0
    MarkerLineWidth = 1;
end

axiscell = iscell(axislimits);

plotsize_struct.width = width;
plotsize_struct.height = height;
plotsize_struct.fontsize = 13;
plotsize_struct.name = 'thesis';
plotsize_struct.caption = '';

for gd=1:NDims
    
    figure

    set(gcf,'Position',[1 1 enbiggen*width/(2*37.7)*scrsz(3) enbiggen*height/30*scrsz(4)])
    

    hold on
    box on
    set(gca,'FontName','Times','FontSize',13);
    
    if ~isempty(XStars)
        NumStar=find(XStars(:,1)==gd);
        sd = shaded_sd(XStars(NumStar,2),YMean(NumStar),YSD(NumStar),Colour);
    end
    
    
    if plot_full
    NumsFull=find(XsFull(:,1)==gd);
    full = plot(XsFull(NumsFull,2),ZsFull(NumsFull),'k','LineWidth',0.5);
    end
    
        Nums=find(XsData(:,1)==gd);
    obs = plot(XsData(Nums,2),ZsData(Nums),'+k','MarkerSize',MarkerSize,'LineWidth',MarkerLineWidth); %'MarkerSize',8
    
    
    if ~isempty(XStars)
        mn = plot(XStars(NumStar,2),YMean(NumStar),'Color',Colour);
    end

    

    
    othercmds();
    
    if length(ylab)>3
        Rotation = 90;
    else
        Rotation = 0;
    end

    ylabel(ylab,'FontName','Times','FontSize',13,'Rotation',Rotation);
    xlabel(xlab,'FontName','Times','FontSize',13);
    title(tit{gd},'FontName','Times','FontSize',13);
    %title([tit,' ',num2str(gd)],'FontName','Times','FontSize',24);
    
    if ~axiscell
        axis(axislimits);
    else
        axis(axislimits{:});
    end
    
    
    
    if NDims>1
        namey = [name,'_',num2str(gd)];
    else
        namey = name;
    end
    if ~nolp
    %postlaprint(gcf,namey,plotsize_struct);
    end
end
if ~nolp
    close all;
end

% ['Chosen position of',sprintf('\n'),'next observation']