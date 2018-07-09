function [sd,mn,obs]=plot_regression(XsFull,YsFull,XStars,YMean,YSD,varargin)

qlegend=false;
Colour=[1 0 0];
args=varargin;

for i=1:length(varargin)/2
    arg=1+(i-1)*2;
    if strcmpi(varargin{arg},'Legend')
        qlegend=true;
        legendloc=varargin{arg+1};
        args=args([1:arg-1,arg+2:end]);
    elseif strcmpi(varargin{arg},'Color')
        Colour=varargin{arg+1};
    end
end


hold on

sd=shaded_sd(XStars,YMean,YSD);
mn=plot(XStars,YMean,'Color',[0 0 0.8],  ...
    'LineStyle', '-', ...
  'LineWidth', 0.75, ...
  'Marker', 'none', ...
  'MarkerSize', 10);
obs=plot(XsFull,YsFull,...
  'MarkerSize',12,...
  'LineStyle', 'none', ...
  'LineWidth', 0.5, ...
  'Marker', '.', ...
  'Color', [0.2 0.2 0.2] ...
  );

if qlegend
    legend([sd,mn,obs],'\pm 1SD','Mean','Observations','Location',legendloc);
end


set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(gca, 'YGrid', 'off');