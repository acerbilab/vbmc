function [best_hypers, best_nll] = fit_hypers_multiple_restart( ...
    X, y, l, init_log_output_scale, number_evals, opt )
% a wrapper for gpml code to train a GP with multiple starting points for
% minimize.m
%
% INPUTS
% - X: inputs
% - y: outputs
% - l: initial guess of log input scales
% - o: (optional) initial guess for log output scale
    
if nargin < 5
    number_evals = 500;
end


% Specify GP Model.
inference = @infExact;
likfunc = @likGauss;
meanfunc = {'meanZero'};
covfunc = @covSEard;
opt_min.length = -number_evals;
opt_min.verbosity = 0;

% Init GP Hypers.
init_hypers.mean = [];
init_hypers.lik = log(1e-4);  % Values go between 0 and 1, so no need to scale.


if size(l,2) == 1
    l = l';
end

init_log_lengthscales = ...
    [l + log(100); l + log(10); l; l - log(10); l - log(100)];
if nargin<4
    init_log_output_scale = log(0.1);
end

for i = 1:size(init_log_lengthscales,1);
    init_hypers.cov = [init_log_lengthscales(i, :) init_log_output_scale]; 
    % Fit the model, but not the likelihood hyperparam (which stays fixed).    
    [gp_hypers{i} fX] = minimize(init_hypers, @gp_fixedlik, opt_min, ...
                                 inference, meanfunc, covfunc, likfunc, ...
                                 X, y);
    nll(i) = fX(end);
end
[best_nll, best_ix] = min(nll);
if opt.print > 1
fprintf('NLL of different lengthscale mins: '); disp(nll);
end
best_hypers = gp_hypers{best_ix};
if any(isnan(best_hypers.cov))
    best_hypers = init_hypers;
    warning('Optimizing hypers failed');
end    
end
