function [posteriors,obs,YStars] = plot_predictive_posterior(XsFull,ZsFull,XsData,ZsData,XStars,rho,YMeans,YSDs,xlab,ylab,tit,YStars,plotsize)
% rho represents the weights over hyperparameter samples, and is a row
% vector of length equal to NSamples. If XStars has NStars rows, YMeans and
% YSDs are both NStars by NSamples, representing the individual mean and
% SDs from each hyperparameter sample.

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
scrsz = get(0,'ScreenSize');

no_YStars = nargin<12;
if nargin<13
    plotsize = 'wide';
end

posteriors_provided = nargin<8 || isempty(YSDs);
if posteriors_provided
    posteriors = YMeans;
end

for gd=1:NDims
    
    switch plotsize
        case 'small'
            figure('Position',[1 1 0.2*scrsz(3) 0.4*scrsz(4)])%0.262*scrsz(4)])
        case 'wide'
            figure('Position',[1 1 0.9*scrsz(3) 0.4*scrsz(4)])
            %figure('Position',[1 1 0.945*scrsz(3) 0.5*scrsz(4)])
            %axis([0 11 0 250])
        case 'smallwide'
            figure('Position',[1 1 0.4725*scrsz(3) 0.3*scrsz(4)])
            %axis([0 11 0 250])
    end
    hold on
    box on
    set(gca,'FontName','Times','FontSize',20);
    
    if ~isempty(XsFull)
        Nums = find(XsFull(:,1)==gd);
    else
        Nums= [];
    end
    
    if ~isempty(XStars)
        load('pinkmap','pinkmap')
        set(gcf,'Colormap',pinkmap);
        
        if no_YStars
            AllYs = [YMeans+3*YSDs,YMeans-3*YSDs];

            YStars = linspace(min(AllYs(:)),max(AllYs(:)),1000);
        end
        
        if ~posteriors_provided
            posteriors = GMM_posterior(XStars,YStars,rho,YMeans,YSDs);
        end
            

        hold on
        %plot(0.29,0.99);
        rangeX = [min(XStars(:,2)) max(XStars(:,2))];
        step = mean(diff(XStars(:,2)));
        rangeY = [min(YStars) max(YStars)];
        imagesc(rangeX+0.5*step,rangeY,posteriors')


        az = 0;
        el = 90;
        view(az, el);
        axis([rangeX rangeY])
        colorbar
        colorbar('YTickLabel',linspace(0,1,6),'FontSize',22)
        colorbar('FontSize',22)
    end
    
    
    if ~isempty(XsFull)
    full = plot(XsFull(:,2),ZsFull(:),'-k','LineWidth',0.5);
    end
    if ~isempty(XsData)
    obs = plot(XsData(:,2),ZsData(:),'+k','MarkerSize',5,'MarkerFaceColor','k','LineWidth',2); %'MarkerSize',8
    end
%     Nums2=find(XsFull(:,1)==(mod(gd,2)+1));
%     obs2 = plot(XsFull(Nums2,2),ZsFull(Nums2),':k','MarkerSize',5, 'LineWidth',2);
    
    if length(ylab)>3
        Rotation = 90;
    else
        Rotation = 0;
    end

    ylabel(ylab,'FontName','Times','FontSize',22,'Rotation',Rotation);
    xlabel(xlab,'FontName','Times','FontSize',22);
    title(tit,'FontName','Times','FontSize',24);
    %title([tit,' ',num2str(gd)],'FontName','Times','FontSize',24);
end

function posteriors = GMM_posterior(XStars,YStars,rho,YMeans,YSDs)
XStars = XStars(:,2);
NXStars = length(XStars);
NYStars = length(YStars);
posteriors = nan(NXStars,NYStars);
[rho_rows,rho_cols] = size(rho);
if rho_rows == size(YMeans,2)
    rho = rho';
    num_hypersamples = rho_rows;
    rho_rows = rho_cols;
else
    num_hypersamples = rho_cols;
end
retrospective = rho_rows==1;



if retrospective
    for ystar_ind = 1:NYStars
        ystar = YStars(ystar_ind);
        posteriors(:,ystar_ind) = normpdf(ystar,YMeans,YSDs)*rho';
    end
else
    for ystar_ind = 1:NYStars
        ystar = YStars(ystar_ind);
        posteriors(:,ystar_ind) = (normpdf(ystar,YMeans,YSDs).*rho)*ones(num_hypersamples,1);
    end
end


