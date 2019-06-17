function gplite_plot(gp,x0)
%GPLITE_PLOT Visualize GP.

% Luigi Acerbi 2019

if nargin < 2; x0 = []; end

[N,D] = size(gp.X);            % Number of training points and dimension
Ns = numel(gp.post);           % Hyperparameter samples
Nx = 100;                      % # grid points per visualization

% Loop over hyperparameter samples
ell = zeros(D,Ns);
for s = 1:Ns
    hyp = gp.post(s).hyp;
    ell(:,s) = exp(hyp(1:D));       % Extract length scales from HYP
end
ellbar = sqrt(mean(ell.^2,2))';      % Mean length scale

lb = min(gp.X) - ellbar;
ub = max(gp.X) + ellbar;

gutter = [.05 .05];
margins = [.1 .01 .12 .01];
linewidth = 1;

if isempty(x0) && D > 1
    [~,idx] = max(gp.y);
    x0 = gp.X(idx,:);
end

for i = 1:D
    ax(i,i) = tight_subplot(D,D,i,i,gutter,margins);
    
    xx_vec = linspace(lb(i),ub(i),ceil(Nx^1.5))';
    
    if ~isempty(x0)
        xx = repmat(x0,[numel(xx_vec),1]);
        xx(:,i) = xx_vec;
    else
        xx = xx_vec;
    end
    [~,~,fmu,fs2] = gplite_pred(gp,xx);
    
    plot(xx_vec,fmu,'-k','LineWidth',1); hold on;
    plot(xx_vec,fmu+1.96*sqrt(fs2),'-','Color',0.8*[1 1 1],'LineWidth',1);
    plot(xx_vec,fmu-1.96*sqrt(fs2),'-','Color',0.8*[1 1 1],'LineWidth',1);
        
    xlim([lb(i),ub(i)]);
    
    set(gca,'TickDir','out','box','off'); 
    if D == 1
        xlabel('x'); ylabel('f(x)');
        scatter(gp.X,gp.y,'ob','MarkerFaceColor','b');        
    else
        if i == 1; ylabel(['x_' num2str(i)]); end
        if i == D; xlabel(['x_' num2str(i)]); end
    end
    
    if ~isempty(x0)
        plot([x0(i),x0(i)],get(gca,'ylim'),'k-','linewidth',linewidth);
    end
end

for i = 1:D
    for j = 1:i-1
        xx1_vec = linspace(lb(i),ub(i),Nx)';
        xx2_vec = linspace(lb(j),ub(j),Nx)';        
        xx_vec = combvec(xx1_vec',xx2_vec')';
        
        xx = repmat(x0,[numel(xx1_vec)*numel(xx2_vec),1]);
        xx(:,i) = xx_vec(:,1);
        xx(:,j) = xx_vec(:,2);
        [~,~,fmu,fs2] = gplite_pred(gp,xx);
        
        for k = 1:2            
            switch k
                case 1
                    i1 = j; i2 = i;
                    mat = reshape(fmu,[Nx,Nx])';
                    ax(i1,i2) = tight_subplot(D,D,i1,i2,gutter,margins);
                    contour(xx1_vec,xx2_vec,mat); hold on;
                case 2
                    i1 = i; i2 = j;
                    mat = reshape(sqrt(fs2),[Nx,Nx]);
                    ax(i1,i2) = tight_subplot(D,D,i1,i2,gutter,margins);
                    contour(xx2_vec,xx1_vec,mat); hold on;
            end

            plot([lb(i2),ub(i2)],[x0(i1),x0(i1)],'k-','linewidth',linewidth);
            plot([x0(i2),x0(i2)], [lb(i1),ub(i1)],'k-','linewidth',linewidth);

            xlim([lb(i2),ub(i2)]);
            ylim([lb(i1),ub(i1)]);
            set(gca,'TickDir','out','box','off');
            
            scatter(gp.X(:,i2),gp.X(:,i1),'.b','MarkerFaceColor','b');
            
        end
        
        if j == 1; ylabel(['x_' num2str(i)]); end
        if i == D; xlabel(['x_' num2str(j)]); end

    end
end

set(gcf,'Color','w');

end

%--------------------------------------------------------------------------
function h=tight_subplot(m, n, row, col, gutter, margins, varargin)
%TIGHT_SUBPLOT Replacement for SUBPLOT. Easier to specify size of grid, row/col, gutter, and margins
%
% TIGHT_SUBPLOT(M, N, ROW, COL) places a subplot on an M by N grid, at a
% specified ROW and COL. ROW and COL can also be ranges
%
% TIGHT_SUBPLOT(M, N, ROW, COL, GUTTER=.002) indicates the width of the spacing
% between subplots, in terms of proportion of the figure size. If GUTTER is
% a 2-length vector, the first number specifies the width of the spacing
% between columns, and the second number specifies the width of the spacing
% between rows. If GUTTER is a scalar, it specifies both widths. For
% instance, GUTTER = .05 will make each gutter equal to 5% of the figure
% width or height.
%
% TIGHT_SUBPLOT(M, N, ROW, COL, GUTTER=.002, MARGINS=[.06 .01 .04 .04]) indicates the margin on
% all four sides of the subplots. MARGINS = [LEFT RIGHT BOTTOM TOP]. This
% allows room for titles, labels, etc.
%
% Will Adler 2015
% will@wtadler.com

if nargin<5 || isempty(gutter)
    gutter = [.002, .002]; %horizontal, vertical
end

if length(gutter)==1
    gutter(2)=gutter;
elseif length(gutter) > 2
    error('GUTTER must be of length 1 or 2')
end

if nargin<6 || isempty(margins)
    margins = [.06 .01 .04 .04]; % L R B T
end

Lmargin = margins(1);
Rmargin = margins(2);
Bmargin = margins(3);
Tmargin = margins(4);

unit_height = (1-Bmargin-Tmargin-(m-1)*gutter(2))/m;
height = length(row)*unit_height + (length(row)-1)*gutter(2);

unit_width = (1-Lmargin-Rmargin-(n-1)*gutter(1))/n;
width = length(col)*unit_width + (length(col)-1)*gutter(1);

bottom = (m-max(row))*(unit_height+gutter(2))+Bmargin;
left   = (min(col)-1)*(unit_width +gutter(1))+Lmargin;

pos_vec= [left bottom width height];

h=subplot('Position', pos_vec, varargin{:});
end