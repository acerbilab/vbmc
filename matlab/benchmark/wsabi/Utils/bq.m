function [mu, Var, kernelVar, lambda] = bq(range, priorMu, priorVar, kernelVar, lambda, alpha, samples, loglikhandle)

% BQ, directly modelling likelihood with GP, no active sampling.
dim = length(kernelVar);
hyp = zeros(1,dim+1);
xx = samples;

% Rename for code brevity.
bb = priorMu;
BB = priorVar;

if isa(loglikhandle,'function_handle')
    lHatD = zeros(length(xx(:,1)),1);
    for i = 1:length(xx(:,1))
        [lHatD(i)] = loglikhandle( xx(i,:) );
    end
else
    lHatD = loglikhandle(:);
end

% Rescale to max.
scaling = max(lHatD);
lHatD = exp(lHatD - scaling);

% Fit the hyperparameters:
hyp(1) = log( lambda );
hyp(2:end) = log( diag(kernelVar)' );

hypLogLik = @(x) logLikGPDim( xx, lHatD, x );
options = optimoptions('fminunc');
options.Display = 'none';
options.GradObj = 'on';
[hyp] = fminunc( hypLogLik, hyp, options );

lambda = exp(hyp(1));
VV = diag(exp(hyp(2:end)));

% Get scaled distance matrix.
xxScaled = xx ./ repmat(sqrt(diag(VV)'),length(xx(:,1)),1);
dist = pdist2_squared_fast(xxScaled,xxScaled);

% Build Gram matrix.
Kxx = lambda^2 * (1/(det(2*pi*VV).^0.5)) * exp(-0.5*dist);
Kxx = Kxx + 1e-8*eye(size(Kxx));

%Expected value of integral:
tmpVar = diag(BB)' + diag(VV)';
xxScaled2 = (xx) ./ repmat(sqrt(tmpVar),length(xx(:,1)),1);
bbScaled2 = bb ./ sqrt(tmpVar);
ww = lambda^2 * det(2*pi*(VV + BB))^-0.5 * exp(-0.5*pdist2_squared_fast(xxScaled2,bbScaled2));
mu = log(ww'*(Kxx \ lHatD)) + scaling;

%Variance of the integral:
Var = log(lambda^2 * det(2*pi*(2 * BB + VV))^-0.5 - ww'*(Kxx \ ww)) + 2*scaling;

kernelVar = diag(exp(hyp(2:end)));

end
