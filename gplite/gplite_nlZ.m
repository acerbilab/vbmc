function [nlZ,dnlZ,post,K_mat,Q] = gplite_nlZ(hyp,gp,hprior)
%GPLITE_NLZ Negative log marginal likelihood for lite GP regression.
%   [NLZ,DNLZ] = GPLITE_INF(HYP,GP) computes the log marginal likelihood 
%   NLZ and its gradient DNLZ for hyperparameter vector HYP. HYP is a column 
%   vector (see below). GP is a GPLITE struct.
%   
%   [NLZ,DNLZ] = GPLITE_INF(HYP,GP,HPRIOR) uses prior over hyperparameters
%   defined by the struct HPRIOR. HPRIOR has fields HPRIOR.mu, HPRIOR.sigma
%   and HPRIOR.nu which contain vectors representing, respectively, the mean, 
%   standard deviation and degrees of freedom of the prior for each 
%   hyperparameter. Priors are generally represented by Student's t distributions.
%   Set HPRIOR.nu(i) = Inf to have instead a Gaussian prior for the i-th
%   hyperparameter. Set HPRIOR.sigma(i) = Inf to have a (non-normalized)
%   flat prior over the i-th hyperparameter. Priors are defined in
%   transformed hyperparameter space (i.e., log space for positive-only
%   hyperparameters).
%
%   [NLZ,DNLZ,POST] = GPLITE_INF(...) also returns a POST structure
%   associated with the provided hyperparameters.
%
%   [NLZ,DNLZ,POST,K_MAT] = GPLITE_INF(...) also returns the computed
%   kernel matrix K_MAT.
%
%   [NLZ,DNLZ,POST,K_MAT,Q] = GPLITE_INF(...) also returns the computed
%   auxiliary matrix Q used for computing derivatives.

if nargin < 3; hprior = []; end

[Nhyp,Ns] = size(hyp);      % Hyperparameters and samples
compute_grad = nargout > 1; % Compute gradient if required

Ncov = gp.Ncov;
Nnoise = gp.Nnoise;
Nmean = gp.Nmean;

if Nhyp ~= (Ncov+Nnoise+Nmean)
    error('gplite_nlZ:dimmismatch','Number of hyperparameters mismatched with dimension of training inputs.');
end
if compute_grad && Ns > 1
    error('gplite_nlZ:NoSampling', ...
        'Computation of the log marginal likelihood is available only for one-sample hyperparameter inputs.');
end

switch nargout
    case {1,2}
        [nlZ,dnlZ] = gplite_core(hyp,gp,1,compute_grad);
    case 3
        [nlZ,dnlZ,post] = gplite_core(hyp,gp,1,compute_grad);
    case 4
        [nlZ,dnlZ,post,K_mat] = gplite_core(hyp,gp,1,compute_grad);
    case 5
        [nlZ,dnlZ,post,K_mat,Q] = gplite_core(hyp,gp,1,compute_grad);
end

% Compute hyperparameter prior if specified
if ~isempty(hprior)
    if compute_grad
        [P,dP] = gplite_hypprior(hyp,hprior);
        nlZ = nlZ - P;
        dnlZ = dnlZ - dP;
    else
        P = gplite_hypprior(hyp,hprior);
        nlZ = nlZ - P;
    end
end

end