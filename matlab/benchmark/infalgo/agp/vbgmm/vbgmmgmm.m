function gmm = vbgmmgmm(vbmodel,method)
%VBGMMGMM Gaussian mixture model (GMM) approximation of variational GMM.
%   GMM = VBGMMEST(VBMODEL) returns a Gaussian mixture model approximation
%   of the variational GMM VBMODEL, by taking the posterior mean over
%   the variational mixture parameters.
%
%   GMM = VBGMMEST(VBMODEL,METHOD) chooses the method for evaluation of the
%   posterior over parameters. METHOD can be 'mean' (default) or 'mode'.
%
%   See also VBGMMFIT.

if nargin < 2 || isempty(method); method = 'mean'; end

nvars = size(vbmodel.U,1);

gmm.w = vbmodel.alpha/sum(vbmodel.alpha);
gmm.Mu = vbmodel.m';
df = zeros(1,1,numel(vbmodel.alpha));
switch lower(method)
    case 'mean'
        df(1,1,:) = (vbmodel.nu - nvars - 1);
        if any(df < 1)
            warning('Moments of the inverse Wishart may not exist.');
            df(df < 1) = 1;
        end
    case 'mode'
        df(1,1,:) = (vbmodel.nu + nvars + 1);
    otherwise
        error('Unknown point estimate. METHOD can be ''mean'' or ''mode''.');
end
        
gmm.Sigma = bsxfun(@rdivide,vbmodel.U,sqrt(df));
gmm.ischol = 1; % SIGMA are stored as Cholesky decompositions

end