function log_likelihood = gp_log_likelihood( log_hypers, X, y, covfunc )
% An attempted reconstruction of the integration problem tackled in the original
% BMC paper.
%
% Uses a SE ARD kernel, so has D + 2 hypers.
%
% David Duvenaud
% February 2012
% =====================

[N,D] = size(X);

inference = @infExact;
likfunc = @likGauss;
meanfunc = {'meanZero'};

num_hyper_evals = size(log_hypers, 1 );
log_likelihood = NaN(num_hyper_evals, 1);
for i = 1:size(log_hypers,1)
    gp_hypers.cov = log_hypers(i, 1:end-1);
    gp_hypers.lik = log_hypers(i, end);
    log_likelihood(i) = -gp(gp_hypers, inference, meanfunc, covfunc, likfunc, X, y);
end
end
