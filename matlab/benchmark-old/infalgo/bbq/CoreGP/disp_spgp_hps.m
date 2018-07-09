function [noise, lambda, w_0, X_c, w_c] = disp_spgp_hps(gp, best_ind, flag)

if ~isfield(gp, 'hypersamples') && isfield(gp, 'logL')
    gp1.hypersamples = gp;
    gp = gp1;
end

if nargin<3
    flag= [];
end

if nargin<2 || isempty(best_ind)
    if isfield(gp.hypersamples, 'logL')
        [logL, best_ind] = max([gp.hypersamples.logL]);
    else
        best_ind = 1;
    end
end
    
best_hypersample = gp.hypersamples(best_ind).hyperparameters;
num_hps = length(best_hypersample);

X_c = gp.hypersamples(best_ind).X_c;
[num_c, num_dims] = size(gp.hypersamples(best_ind).X_c);
    
if isfield(gp, 'w0_inds')
    noise_ind = gp.logNoiseSDPos;
    lambda_ind = gp.lambda_ind;
    w0_inds = gp.w0_inds;
else
    noise_ind = 1;
    w0_inds = 2:(num_dims+1);
    lambda_ind = num_dims+2;
end

noise = exp(best_hypersample(noise_ind));
lambda = exp(best_hypersample(lambda_ind));
w_0 = exp(best_hypersample(w0_inds));


tw_c = ...
    exp(gp.hypersamples(best_ind).log_tw_c);
w_c = bsxfun(@plus, tw_c, 0.5*w_0);


    
if nargout == 0
    if ~strcmpi(flag,'no_logL')
        fprintf('log-likelihood of %g for ', logL);
    end
    fprintf('hyperparameters:\n');
    
    fprintf('noise SD:\t%g\n', ...
        noise);
    fprintf('output scale:\t%g\n', ...
        lambda*(prod(2*pi*w_0))^(-1/4));
    fprintf('w_0:\t[');
    fprintf(' %f', w_0);
    fprintf(']\n');
    
    for c = 1:num_c
        fprintf('w_c(%d): [', c);
        fprintf(' %f', w_c(c,:));
        fprintf(']\t X_c(%d): [', c);
        fprintf(' %f', X_c(c,:));
        fprintf(']\n');
    end
end
    
