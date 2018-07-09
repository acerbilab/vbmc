% gridLegend : plots a legend in a multi column format
%
% On a plot with a lot of traces the standard legend will often scroll off the bottom or the side of the figure,
% this function is intended to overcome this by allowing the user to define a multi column format for the
% legend.
% 
%  Usage : legHdl = gridLegend(hdl,nCols,gKey,'parameter_name','parameter_value',...);
%
%  Inputs :  hdl -  a vector of graphic handles of the plotted data 
%               nCols - the number of columns in the legend. (defaults to 2 if not defined)
%               gKey - a cell array of strings to display which should match the number of graphic handles in hdl (optional)
%               'parameter_name','parameter_value' - any parameter name pairs applicable to the legend
%
%  Output : legHdl - a vector of graphic handles for the legend generated.
%
%  Notes :
%
%  By default the legend will be arranged vertically, ie the second legend trace is placed under the first
%  trace, filling the first column before moving onto the next. If you use the 'Orientation','Horizontal'
%  parameter pair then the traces will be arranged horizontally, so the second trace will be in the second
%  column and so on to the end of the row, then moving down to the next row.
% 
%  The parameter name / value pairs are passed over to the legend function so you can set things
%  like 'Fontsize',8 etc.
%
%  The parameter / name pair of location can be used to place the legend - available options are
%      'location','bestoutside' : for 2 column legend uses 'eastoutside', more than 2 it uses 'southoutside'
%      'location','north' : places the legend centrally at the top of the axes
%      'location','northoutside' : places the legend centrally at the top of the figure
%      'location','eastoutside' : places the legend on the right of the figure
%      'location','southoutside' : places the legend centrally at the bottom of the figure
%      'location','westoutside' : places the legend on the left of the figure
%
%  By default this is set to 'bestoutside'
%
%  This will work on x-y plots, bar and errorbar charts.
%
%  With many traces it's sometimes difficult to work out which trace the legend is referring to. From the
%  MATLAB file exchange I've used a function called 'clickableLegend' which allows you to switch the plotted
%  data on and off by clicking on the Legend key. This function will use clickableLegend if it can find it
%  in your MATLAB path, otherwise it will revert back to the standard legend function.
%
% Examples : 
%
% This will plot a vertical legend with two columns
%        gridLegend(hdl,2)
%
% This will plot a horizontal legend, 8 traces per row, and fontsize set to 8 points.
%        gridLegend(hdl,8,gKey,'Orientation','Horizontal','Fontsize',8)
% 
% This will plot a vertical legend on the left hand side with 2 columns with the box switched off.
% I've fiddled with the XColor and YColor otherwise we end up with black lines for the X and Y axis.
%        gridLegend(hdl,2,gKey,'location','westoutside','Fontsize',8,'Box','off','XColor',[1 1 1],'YColor',[1 1 1])
% 

%  Adrian Cherry
%  adrian.cherry@baesystems.com
%  13/1/2011
%

function legend_h = gridLegend(hdl,gd,varargin)

% set default column number to 2 if not defined
if nargin < 2,
    gd = 2;
end

% test if the user has clickableLegend available, otherwise default to the standard legend.
if exist('clickableLegend','file'),
    fLegend = @clickableLegend;
else
    fLegend = @legend;
end

% pull out the orientation parameter so we can work out which way to go, across or down when moving the 
% legend traces around but create the legend initially in vertical mode. Also pull out location if defined.
location='bestoutside';
orient='vertical';

% identify the start position for the parameter name pairs, if there are an odd number of varargin
% inputs then I'm guessing that the first one is a cell array of data labels, so start ofn the second input.
st=1;
if mod(length(varargin),2),
    st = 2;
end
for i=st:2:length(varargin),
    switch lower(varargin{i}); 
        case {'orientation'}
            orient=varargin{i+1};
            varargin(i+1)={'vertical'};
        case {'location'}
            location=lower(varargin{i+1});
    end   
end

% for the bestoutside option if its 2 columns stick it on the right, else put it at the bottom.
if strcmpi(location,'bestoutside'),
    if gd==2,
        location='eastoutside';
    else
        location='southoutside';
    end
end

% create the normal legend supplying it all the parameter name value pairs given in the function call
[legend_h,object_h] = fLegend(hdl,varargin{:});

% if only one column then bail out now - nothing else to do here. Although it might seem daft to call
%gridLegend with one column this does allow the user the flexibility of always calling gridLegend and their
% code can adapt the number of columns required in which case one column might be a valid input.
if gd < 2,
    return
end

% work out how many traces per column in the new format.
numlines = length(hdl);
numpercolumn = ceil(numlines/gd);

% if we don't get enough legend objects than something has gone wrong generating the legend..
if length(object_h) < 2*numlines,
    warning('Sorry problems generating the standard legend - not enough labels were generated')
    return
end

%get old width, new width and scale factor
pos = get(legend_h, 'position');
width = gd*pos(3);
rescale = pos(3)/width;

% get some old values so we can scale everything later
% if it's x-y plot then there are three objects per plotted line: a label, a line and a marker.
hdlMarker = object_h(numlines+1);
if isprop(hdlMarker, 'xdata'),
    plotType = 'xyPlot';
    xdata = get(hdlMarker, 'xdata');
    dx = xdata(2)-xdata(1);
    di = 2;
% if it's a bar chart then only 2 objects per plotted line: a labela and a colour patch.
elseif isprop(get(hdlMarker,'children'), 'xdata')
    plotType = 'barChart';
    xdata = get(get(hdlMarker,'children'), 'xdata');
    ydata = get(get(hdlMarker,'children'), 'ydata');
    dx = xdata(3)-xdata(1);
    dy = ydata(2)-ydata(1);
    di = 1;
% if the marker objects is a hggroup then one possibility is an errorbar plot
elseif strcmp(get(get(hdlMarker,'children'), 'Type'),'hggroup')
    plotType = 'errorBar';
    ec = get(get(hdlMarker,'children'), 'children');
    xdata = get(ec(2), 'xdata');
    dx = xdata(2)-xdata(1);
    di = 1;
else
    error('Can''t work out what sort of legend we''ve got - sorry bailing out');
end
    
% we'll use these later to align things appropriately
sheight = 1/numpercolumn;                  % height between data lines
height = 1-sheight/2;                            % height of the box. Used to top margin offset
line_width = dx*rescale;                        % rescaled linewidth to match original
spacer = xdata(1)*rescale;                    % rescaled spacer used for margins


% put the legend on the upper left corner and increase to required width.
loci = get(gca, 'position');
set(legend_h, 'position', [loci(1) pos(2) width pos(4)]);


% for each trace and label in the legend we need to define the new column and row position
col = -1;
position=-1;
for i=1:numlines,
    % for horizontal legends increment the column on each loop and add one to the row when we reach the end
    if strcmpi(orient,'horizontal'),
        if mod(i,gd)==1,
            position = position+1;
        end
        col = mod(i,gd)-1;
        if col == -1,
             col = gd-1;
        end
    % for vertical legends increment the row on each loop and add one to the column when we reach the bottom
    else
        if numpercolumn==1 || mod(i,numpercolumn)==1,
            col = col+1;
        end

        position = mod(i,numpercolumn)-1;
        if position == -1,
             position = numpercolumn-1;
        end   
    end
    
    % in the legend list of objects the first handle is the label, the second is the label and the third is the marker
    if i==1
        linenum = i+numlines;
    else
        linenum = linenum+di;
    end
    labelnum = i;
    
    
    % realign the lines and markers for the x-y plots
    switch plotType;
        case 'xyPlot'
            set(object_h(linenum), 'ydata', [height-position*sheight height-position*sheight]);
            set(object_h(linenum), 'xdata', [col/gd+spacer col/gd+spacer+line_width]);
            set(object_h(linenum+1), 'ydata', height-position*sheight);
            set(object_h(linenum+1), 'xdata', col/gd+spacer+line_width/2);
        % or move the color patch and label for the bar charts 
        case 'barChart'
            hdb=get(object_h(linenum),'children');
            x1 = col/gd+spacer;
            x2 = x1 + line_width;
            y1 = height-position*sheight-dy*numlines/(2*numpercolumn);
            y2 = y1 + dy*numlines/numpercolumn; % need to make the colour patch bigger so that when rescaled it returns to a sensible size.
            set(hdb,'xdata',[x1 x1 x2 x2 x1]');
            set(hdb,'ydata',[y1 y2 y2 y1 y1]');
        case 'errorBar'
            hdlEb = get(get(object_h(linenum),'children'), 'children');
            set(hdlEb(2), 'ydata', [height-position*sheight height-position*sheight]);
            set(hdlEb(2), 'xdata', [col/gd+spacer col/gd+spacer+line_width]);
            set(hdlEb(1), 'ydata', height-position*sheight);
            set(hdlEb(1), 'xdata', col/gd+spacer+line_width/2);
    end
    % move the legend label to the new position.
    set(object_h(labelnum), 'position', [col/gd+spacer*2+line_width height-position*sheight]);
      
end

% resize the data aspect ratio to match the new shape.
set(legend_h,'dataaspectratio',[width/gd gd/numpercolumn 1]);

% get the axes sizes and tightInset values so we can work out the padding required for the axis labels.
ti = get(gca,'TightInset');
axp = get(gca,'Position');

% calculate the required position for the legend and current axes so that there is space for it.
switch location,
    case {'north'}
        % top location
        np=[(1-width)/2 1-numpercolumn/numlines width pos(4)*numpercolumn/numlines]; 
    case {'southoutside'}
        % middle bottom location
        np=[(1-width)/2 0 width pos(4)*numpercolumn/numlines]; 
        axn = np(4) + ti(2);
        % define the new axis postion and apply
        axp(4)=axp(4) - axn + axp(2);
        axp(2)=axn;
    case {'eastoutside'}
        % right hand side location
        np=[0.995-width numpercolumn/numlines/2 width pos(4)*numpercolumn/numlines];
        axp(1)=axp(1)/2;
        axp(3)=0.995-width-axp(1)-ti(3);
    case {'westoutside'}
        % left hand side location
        np=[0.005 numpercolumn/numlines/2 width pos(4)*numpercolumn/numlines];
        axp(1)=0.01+width+ti(1);
        axp(3)=0.99-axp(1)-ti(3);
    case {'northoutside'}
        % middle top location
        np=[(1-width)/2 1-numpercolumn/numlines width pos(4)*numpercolumn/numlines]; 
        % define the new axis postion and apply
        axp(4)=1 - np(4) - ti(4)- axp(2);
    otherwise
        % no changes applied to the axis sizing
        % left hand side location
        np=[0.005 numpercolumn/numlines/2 width pos(4)*numpercolumn/numlines];
        
end

% finally move the legend and update the axes
set(legend_h, 'Position',np);
set(gca,'Position',axp);

% also disable legend listeners, printpreview redrew all the legend and messed everything up
% fortunately thanks to undocumented matlab website worked out how to make it static.
LL = get(gca,'ScribeLegendListeners');
set(LL.fontname,'enabled','off');
set(LL.fontsize,'enabled','off');
set(LL.fontweight,'enabled','off');
set(LL.fontangle,'enabled','off');
set(LL.linewidth,'enabled','off');
set(gca,'ScribeLegendListeners',LL);


return
