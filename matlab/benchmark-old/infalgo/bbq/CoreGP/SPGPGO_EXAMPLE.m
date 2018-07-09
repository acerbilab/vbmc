n=11;
lower_bound = [-5 0];
upper_bound = [10 15];
x0 = rand(n,2);
x0 = x0(end,:);
x0 = x0.*(upper_bound - lower_bound) + lower_bound;
%mean([lower_bound; upper_bound]);

% three global optima:  (-pi, 12.275), (pi, 2.275), (9.42478, 2.475), where
% branin = 0.397887

% start optimizing
f = @(x) branin(x);

opt.verbose = false;
opt.plots = true;
opt.total_time = 180;
opt.optim_time = 80;
opt.num_c = 50;
opt.num_hypersamples = n;
opt.save_str = 'branin_example2';
opt.num_retrains = 50;
opt.pool_open = true;
[minimum, minimum_location, X_data, y_data, gp, quad_gp] = ...
    spgpgo(f, x0, lower_bound, upper_bound, opt);

num_dims = size(X_data,2);
for dim  = 1:num_dims
    
    XStar = repmat(minimum_location,1000,1);
    XStar(:,dim) = linspace(lower_bound(dim),upper_bound(dim),1000);

    [YMean,YSD] = ...
        predict_spgp(XStar, gp, quad_gp);

    figure
    params.x_label = ['x',num2str(dim)];
    params.width = 20;
    params.height = 10;
    params.legend_location = 'EastOutside';
    gp_plot(XStar(:,dim), YMean, YSD, X_data(:,dim), y_data, ...
        [], [], params);
%     ylim([min(y_data), max(y_data)]);
%     xlim([min(X_data(:,dim)), max(X_data(:,dim))]);
%    saveas(gcf,['x',num2str(dim),'.png'])
end

%input_importance(gp, X_data)

figure
plot3(X_data(:,1), X_data(:,2), y_data,'.')