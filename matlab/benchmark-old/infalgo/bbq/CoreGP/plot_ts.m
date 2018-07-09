function handle=plot_ts(XsFull,YsFull,XStars,YMean,YSD)

NDims=length(unique(XsFull(:,1)));

JColours=copper;
handle=figure;
hold all
for gd=1:NDims
    Colour=JColours(ceil(64*(gd-0.5)/NDims),:);
    NumStar=find(XStars(:,1)==gd);
    shaded_sd(XStars(NumStar,2),YMean(NumStar),YSD(NumStar),Colour);
end
for gd=1:NDims
    Colour=JColours(ceil(64*(gd-0.5)/NDims),:);
    Nums=find(XsFull(:,1)==gd);
    plot(XsFull(Nums,2),YsFull(Nums),'+','Color',Colour,'LineWidth',2);
    NumStar=find(XStars(:,1)==gd);
    plot(XStars(NumStar,2),YMean(NumStar),'Color',Colour);
end