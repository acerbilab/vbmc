

close all
clear

N = 5;
% observation inputs
X_data = [1:100]';
X_data = repmat(X_data,1,N);
% observation outputs
y_data = sin(X_data(:,1)/5);%.*cos(X(:,1)/5).^2;

% inputs at which to make predictions
X_star = [-50:150]';
X_star = repmat(X_star,1,N);

optim_time = 60;

% training
gp = train_gp('sqdexp', 'constant', [], X_data, y_data, optim_time);

% testing
[t_mean,t_sd]=predict_gp(X_star, gp);

% the real values at X_star
real_star = sin(X_star(:,1)/5);
   
for i = 1: size(X_data,2)
    figure
    params.x_label = ['x_',num2str(i)];
    gp_plot(X_star(:,i), t_mean, t_sd, X_data(:,i), y_data, X_star(:,i), real_star, params);
end

figure
correlation_plot(t_mean, t_sd, real_star)
