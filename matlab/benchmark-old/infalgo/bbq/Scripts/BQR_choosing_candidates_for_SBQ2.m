
close all
scrsz = get(0,'ScreenSize');
width=7.5;
height=3;



% mean = 2*(0.5-rand(1,2));
% cov = factor*rand(2);
% cov = cov'*cov;

mean = [0 0];

cov = [0.15    0.3
    0.3    0.9];

% numer_weights = [3 0.2];
% denom_weights = [0.1 0.08];
% numer_weights = rand(1,2);
% denom_weights = rand(1,2);

numer_weights = [0.5271    0.4574];
    
denom_weights = [0.8754    0.5181];

density = @(xs) mvnpdf(xs,mean,cov)
ratio = @(xs) (exp(xs)*numer_weights')./(exp(xs)*denom_weights');


% density = @(xs) mvnpdf(log(xs),log(mean),cov).*(xs(:,1).*xs(:,2)).^(-1);
% ratio = @(xs) (xs*numer_weights')./(xs*denom_weights');

f=@(xs) density(xs).*ratio(xs);

factor=1;
num = 100;
vec = linspace(-factor*3,factor*3,num)';

[X,Y] = meshgrid2d(vec+ mean(1),vec+ mean(2));

xs = [X(:),Y(:)];

f_vec = f(xs);
density_vec = density(xs);
ratio_vec = ratio(xs);

fs = reshape(f_vec,num,num);
densities = reshape(density_vec,num,num);
ratios = reshape(ratio_vec,num,num);

figure
hold on
c1 = contour(X,Y,densities);
c2 = contour(X,Y,ratios,'--');
set(gcf,'renderer','zbuffer');

xlabel('$\tilde{r}_i$')
ylabel('$\tilde{r}_j$','Rotation',0)
axis([-2 2 -2 2])

set(gcf,'units','centimeters','Position',[0 0 width height])
legend_handle = legend('$\p{\tvr}{\tvr_s}$','$\varrho(\tvr)$','Location',[0.9 0.6 0.2 0.1])
set(legend_handle, 'EdgeColor', [0.99 0.99 0.99])
set(gca, 'TickDir', 'out')
matlabfrag('~/Documents/SBQ/linearisation_r')


ratio = @(xs) (exp(xs)*(0.2*[0.1 0.17])');%./(exp(xs)*denom_weights');

ratio_vec = ratio(xs);
ratios = reshape(ratio_vec,num,num);

figure
hold on
c1 = contour(X,Y,densities);
c2 = contour(X,Y,ratios,'--');
set(gcf,'renderer','zbuffer');

xlabel('$\tilde{q}_i$')
ylabel('$\tilde{q}_j$','Rotation',0)
axis([-2 2 -2 2])

set(gcf,'units','centimeters','Position',[0 0 width height])
legend_handle2 = legend('$\p{\tvq}{\tvq_s}$','$\varrho(\tvq)$','Location',[0.9 0.6 0.2 0.1])
set(legend_handle2, 'EdgeColor', [0.99 0.99 0.99])
set(gca, 'TickDir', 'out')
matlabfrag('~/Documents/SBQ/linearisation_q')