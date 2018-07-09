

close all
clear

N = 1;
% observation inputs
X_data = linspace(1,100, 50)';
X_data = repmat(X_data,1,N);
% observation outputs

f = @(X) sin(X(:,1)/5).*(1+1/50^2*(X(:,1)-50).^2);
y_data = f(X_data);

% inputs at which to make predictions
X_star = [-50:150]';
X_star = repmat(X_star,1,N);

opt.optim_time = 60;
opt.parallel = false;
opt.num_hypersamples = 5;
opt.noiseless = false;
opt.verbose = true;
opt.plots = false;

% training
tic
gp = lw_train_gp('sqdexp', 'constant', [], X_data, y_data, opt);
toc

% testing
[t_mean,t_sd] = predict_gp(X_star, gp);

% the real values at X_star
real_star = f(X_star);
   
for i = 1: size(X_data,2)
    figure
    params.x_label = ['x_',num2str(i)];
    gp_plot(X_star(:,i), t_mean, t_sd, X_data(:,i), y_data, X_star(:,i), real_star, params);
end

figure
correlation_plot(t_mean, t_sd, real_star)
