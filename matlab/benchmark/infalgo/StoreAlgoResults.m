function [history,post] = StoreAlgoResults(probstruct,post,Niter,X,y,mu,vvar,Xiter,yiter,TotalTime)
%STOREALGORESULTS Store results of running an inference algorithm.

history = infbench_func(); % Retrieve history
% history.scratch.output = output;
history.TotalTime = TotalTime;
history.Output.X = X;
history.Output.y = y;
if ~probstruct.AddLogPrior      % y stores log posteriors, so add prior now
    lnp = infbench_lnprior(history.Output.X,probstruct);
    history.Output.y = history.Output.y + lnp;
end

post.lnZ = mu(end);
post.lnZ_var = vvar(end);
X_train = history.Output.X;
y_train = history.Output.y;
[post.gsKL,post.Mean,post.Cov,post.Mode] = computeStats(X_train,y_train,probstruct);

% Return estimate, SD of the estimate, and gauss-sKL with true moments
N = history.SaveTicks(1:Niter);
history.Output.N = N(:)';
history.Output.lnZs = mu(:)';
history.Output.lnZs_var = vvar(:)';

for iIter = 1:Niter
    if isempty(Xiter) || isempty(yiter)
        X_train = history.Output.X(1:N(iIter),:);
        y_train = history.Output.y(1:N(iIter));
    else
        X_train = Xiter{iIter};
        y_train = yiter{iIter};
    end
    [gsKL,Mean,Cov,Mode] = computeStats(X_train,y_train,probstruct);
    history.Output.Mean(iIter,:) = Mean;
    history.Output.Cov(iIter,:,:) = Cov;
    history.Output.gsKL(iIter) = gsKL;
    history.Output.Mode(iIter,:) = Mode;    
end


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [gsKL,Mean,Cov,Mode] = computeStats(X,y,probstruct)
%COMPUTE_STATS Compute additional statistics.
    
% Compute Gaussianized symmetric KL-divergence with ground truth
gp.X = X;
gp.y = y;
gp.meanfun = 4; % Negative quadratic mean fcn

Ns_moments = 2e4;
xx = gplite_sample(gp,Ns_moments);
Mean = mean(xx,1);
Cov = cov(xx);
[kl1,kl2] = mvnkl(Mean,Cov,probstruct.Post.Mean,probstruct.Post.Cov);
gsKL = 0.5*(kl1 + kl2);

Mode = gplite_fmin(gp,[],1);    % Max flag - finds maximum

end