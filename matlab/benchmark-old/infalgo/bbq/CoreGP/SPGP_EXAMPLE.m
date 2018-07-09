clear

% make up some data
num_data = 1000;
num_dims = 2;

scales = ones(1,num_dims);
means = zeros(1,num_dims);
X_data = bsxfun(@plus,rand(num_data, num_dims)*diag(scales),means);
y = @(X) sin(5*X(:,1))./X(:,1) + 0.4*X(:,2);
y_data = y(X_data);

% the number of basis functions to use
opt.num_c = 50;

% the number of seconds to allow for training -- this influences the degree
% of exploitation of hyperparameter space
opt.optim_time = 50;

% the number of hyperparameter samples to use -- this influences the degree
% of exploration of hyperparameter space
opt.num_hypersamples = 10;

% print diagnostic information to screen?
opt.verbose = false;
opt.plots = false;

% the mean function to use for 
opt.mean_fn = 'affine';

% If gp is non-empty, train_spgp will use its best hyperparameters as an
% initial guess for the training process
gp = [];

% train the gp
[gp, quad_gp] = train_spgp(gp, X_data, y_data, opt);

save test_spgp

% define the points X_star at which to make predictions. real_star are the
% real values at those points, for evaluation purposes. 
X_star = bsxfun(@plus,rand(1000,num_dims)*diag(scales),means);
real_star = y(X_star);

% test the gp
[m, sd] = predict_spgp(X_star, gp, quad_gp);
[X_c, y_c] = spgp_centres(gp, quad_gp);

% make some plots
for i = 1: size(X_data,2)
    figure(i)
    clf
    params.x_label = ['x_',num2str(i)];
    spgp_plot(X_star(:,i), m, sd, X_c(:,i), y_c, ...
        X_data(:,i), y_data, X_star(:,i), real_star, params);
end

figure(size(X_data,2)+1)
clf
correlation_plot(m, sd, real_star)
% 
[RMSE,normed_RMSE]=performance(X_star,m,sd,X_star,real_star)