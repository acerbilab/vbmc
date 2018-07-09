clear
xd = (1:100)';
xstar = -5;
x = [xstar;xd];

K = exp(-0.5*(bsxfun(@minus, x', x)).^2);
y = mvnrnd(zeros(length(x),1), K)';

yd = y(2:end);
yst = y(1);

gp = set_gp('matern', 'constant', [], xd, yd, 1);

M = 100;

m = nan(M,1);
C = nan(M,1);
logL = nan(M, 1);
log_input_scales = linspace(-2,1,M);
input_scales = exp(log_input_scales);
for i = 1:M
    gp.hypersamples.hyperparameters(2) = input_scales(i);
    gp = revise_gp(xd, yd, gp, 'overwrite', [], 1);
    logL(i) = gp.hypersamples.logL;
    [m(i),C(i)] = posterior_gp(xstar, gp, 1);
end

sd = sqrt(C);



clf
scatter(log_input_scales, m, [], exp(logL), 'LineWidth', 1.5)
colorbar
hold on
scatter(log_input_scales, sd, [], exp(logL), 'LineWidth', 1.5)
line([min(log_input_scales), max(log_input_scales)],[yst, yst], 'Color','k')