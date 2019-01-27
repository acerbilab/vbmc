%GPLITE_DEMO Demo script with example usage for the GPLITE toolbox.

% Create example data in 1D
x = linspace(-5,5,11)';
y = sin(x);
 
hyp0 = [];          % Starting hyperparameter vector for optimization
Ns = 8;             % Number of hyperparameter samples
meanfun = 'const';  % GP mean function
hprior = [];        % Prior over hyperparameters
options = [];       % Additional options

% Train GP on data
[gp,hyp,output] = gplite_train(hyp0,Ns,x,y,meanfun,hprior,[],options);

hyp             % Hyperparameter samples

xstar = linspace(-15,15,200)';   % Test points

% Compute GP posterior predictive mean and variance at test points
[ymu,ys2,fmu,fs2] = gplite_pred(gp,xstar);

% Plot data and GP prediction
close all;
figure(1); hold on;
plot(xstar,fmu+sqrt(fs2),'-','Color',0.8*[1 1 1],'LineWidth',1);
plot(xstar,fmu-sqrt(fs2),'-','Color',0.8*[1 1 1],'LineWidth',1);
plot(xstar,fmu,'k-','LineWidth',1);
scatter(x,y,'ob','MarkerFaceColor','b');

xlabel('x');
ylabel('f(x)');
set(gca,'TickDir','out');
box off;
set(gcf,'Color','w');



