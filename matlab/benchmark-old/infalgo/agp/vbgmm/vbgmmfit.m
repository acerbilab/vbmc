function [vbmodel,labels,L,removed] = vbgmmfit(X,m,prior,options)
%VBGMMFIT Variational Bayes fit of Gaussian mixture model.
%   VBMODEL = VBGMMFIT(X,M) fits a variational Gaussian mixture model to 
%   D-by-N data matrix X. Columns of X correspond to data points, rows 
%   correspond to variables. The initialization parameter M can be:
%      * an integer that specifies the maximum number of mixture components;
%      * a 1-by-N label vector that specifies intial cluster assignments 
%        for the data points; 1 <= M(i) <= K specifies the cluster the i-th 
%        data point belongs to.
%      * a model structure (as per VBMODEL).
%   The returned VBMODEL is a trained model structure.
%   VBGMMFIT ignores data points containing Inf or NaN values.
%
%   VBMODEL = VBGMMFIT(X,M,PRIOR) specifies a prior structure PRIOR for
%   the variational inference. (documentation needed)
%
%   VBMODEL = VBGMMFIT(X,M,PRIOR,OPTIONS) replaces the default parameters
%   with values in the structure OPTIONS. VBGMMFIT uses these options:
%
%      OPTIONS.Display defines the level of display. Accepted values for
%      Display are 'iter', 'notify', 'final', and 'off' for no display. The 
%      default value of Display is 'off'. 
% 
%      (documentation needed)
%
%   [VBMODEL,LABELS] = VBGMMFIT(...) returns cluster labels for the trained
%   Gaussian mixture model. 
%
%   [VBMODEL,LABELS,L] = VBGMMFIT(...) returns the variational lower bound.
%
%   [VBMODEL,LABELS,L,REMOVED] = VBGMMFIT(...) returns a 1-by-N logical 
%   array whose elements are true for the data points of X that were
%   excluded from the analysis.
%
%   See also VBGMMPDF, VBGMMPRED, VBGMMRND. 

% Reference: Christopher M. Bishop, Pattern Recognition and Machine
% Learning, Springer-Verlag, New York, 2006 (Chapter 10).

% Author:   Luigi Acerbi
% Email:    luigi.acerbi@gmail.com
%
% This toolbox was inspired by "VB inference for GMM" toolbox by Mo Chen:
% http://www.mathworks.com/matlabcentral/fileexchange/35362-variational-bayesian-inference-for-gaussian-mixture-model

% Bounds are only partially supported


% Check priors for constrained variables

if nargin < 2; m = []; end
if nargin < 3; prior = []; end
if nargin < 4; options = []; end
if isempty(m); m = 100; end

% Default options
defopts.Display     = 'notify';     % Display
defopts.TolBound    = 1e-8;         % Minimum relative improvement on variational lower bound
defopts.Niter       = 2000;         % Maximum number of iterations
defopts.Nstarts     = 5;            % Number of runs
defopts.TolResponsibility = 0.5;    % Remove components with less than this total responsibility
defopts.ClusterInit = 'kmeans';     % Initialization of VB (methods are 'rand' and 'kmeans')

% Assign default options if not defined
for f = fields(defopts)'
    if ~isfield(options,f{:}) || isempty(options.(f{:}))
        options.(f{:}) = defopts.(f{:});
    end
end

% Display options
switch options.Display
    case {'notify','notify-detailed'}
        trace = 2;
    case {'none', 'off'}
        trace = 0;
    case {'iter','iter-detailed'}
        trace = 3;
    case {'final','final-detailed'}
        trace = 1;
    otherwise
        trace = 1;
end

tol = options.TolBound;
maxiter = options.Niter;
starts = options.Nstarts;

% Set bounds ([-Inf,Inf] if not specified)
LB = []; UB = [];
if ~isempty(prior) && isfield(prior,'LB'); LB = prior.LB(:); end
if ~isempty(prior) && isfield(prior,'UB'); UB = prior.UB(:); end

prior.mean_orig = mean(X,2);
prior.var_orig = var(X,[],2);

% Initialize prior
[~,nold] = size(X);
[vbmodel0,X,removed] = vbinit(X,m,prior,LB,UB);
if any(removed) && trace > 1
    fprintf('Removed %d points in X on or outside bounds, or with Inf/NaN values.\n',sum(removed));
end
[~,n] = size(X);

kold = Inf;
L = -Inf;
for irun = 1:starts    
    if trace > 1
        fprintf('              Variational         Relative          Mixture\n    Iter      Lower Bound      Improvement       Components\n');
        displayFormat = ' %7.0f     %12.6g     %12.6g     %12g\n';
    end
    Liter = -Inf(1,maxiter);
    
    % Reset model
    vbtemp = vbmodel0;    
    if isempty(vbtemp.R)    % Initialize labels
        k = size(vbtemp.R,2);
        switch lower(options.ClusterInit)
            case 'rand'     % Random cluster assignment
                labels = randi(k,[1,n]);
            case 'kmeans'
                kmeansopt.Display = 'off';
                kmeansopt.Preprocessing = 'normalize';
                labels = fastkmeans(X',k,kmeansopt);
            otherwise
                error('OPTIONS.ClusterInit can be ''rand'' for random cluster assignment or ''kmeans'' for K-means clustering.');
        end
        vbtemp.R = full(sparse(1:n,labels,1,n,k,n));                
    end
    
    vbtemp = vbmaximize(X,vbtemp);    
    if trace > 1; fprintf(displayFormat,1,Liter(1),0,size(vbtemp.m,2)); end

    % Main loop
    for iter = 2:maxiter
        vbtemp = vbexpect(X,vbtemp);
        vbtemp = vbmaximize(X,vbtemp);
        Liter(iter) = vbbound(X,vbtemp)/n;
        impro = abs((Liter(iter)/Liter(iter-1)-1));    % Relative improvement
        k = size(vbtemp.m,2);
        if trace > 2 || trace > 1 && k ~= kold
            fprintf(displayFormat,iter,Liter(iter),impro,k);
        end
        kold = k;
        if impro < tol; break; end
        vbtemp = vbprune(vbtemp,options.TolResponsibility);
    end
    
    if Liter(iter) > L
        vbmodel = vbtemp;
        L = Liter(iter);
    end
end

if starts == 0
    vbmodel = vbmodel0;
    L = vbbound(X,vbmodel);
end

% Remove empty components
vbmodel = vbprune(vbmodel,options.TolResponsibility);

vbmodel.prior = rmfield(vbmodel.prior,'mean_orig');
vbmodel.prior = rmfield(vbmodel.prior,'var_orig');

% warning('remove R!');

% L = L(2:iter);
if nargout > 1
    labels = zeros(1,nold);
    [~,labels(~removed)] = max(vbmodel.R,[],2);
end

end


%--------------------------------------------------------------------------
function [vbmodel,X,removed] = vbinit(X,m,prior,LB,UB)
%VBINIT Initialize model and prior for variational inference.

[d,n] = size(X);
if isstruct(m)  % init with a model
    vbmodel = m;
    % Take bounds from model if not specified
    if isempty(LB); LB = vbmodel.prior.LB; end
    if isempty(UB); UB = vbmodel.prior.UB; end
    % Take prior from model, if no alternative prior is specified
    if isempty(prior); prior = vbmodel.prior; end    
elseif numel(m) == 1  % random initialization of k
    k = m;
    vbmodel.R = zeros(0,k); % Do this later
elseif all(size(m)==[1,n])  % initialization with labels
    label = m;
    k = max(label);
    vbmodel.R = full(sparse(1:n,label,1,n,k,n));
else
    error('Wrong initialization argument M. Digit ''help vbgmmfit'' for usage information.');
end

% Set bounds
if isempty(LB); LB = -Inf(d,1); end
if isempty(UB); UB = Inf(d,1); end    
prior.LB = LB;
prior.UB = UB;    

% width = (UB-LB)/10;
% width(isinf(width)) = 1;

% Remove points outside bounds or with Infs or NaNs
removed = any(bsxfun(@le, X, LB) | bsxfun(@ge, X, UB),1) | any(~isfinite(X),1);
if any(removed)
    X(:,removed) = [];
    if isfield(vbmodel,'R') && ~isempty(vbmodel.R)
        vbmodel.R(:,removed) = [];
    end
    [d,n2] = size(X);
end

% First run, reparametrize space if needed
X = vbtransform(X,LB,UB,'dir');

% Fill prior fields with default values
if ~isfield(prior,'alpha') || isempty(prior.alpha); prior.alpha = 1; end
if ~isfield(prior,'beta') || isempty(prior.beta); prior.beta = 1; end
if ~isfield(prior,'nu') || isempty(prior.nu); prior.nu = d+1; end
% Empirical Bayes approach for mean and scale matrix
if ~isfield(prior,'m') || isempty(prior.m)
    prior.m = mean(X,2);
    prior.m_orig = prior.mean_orig;
elseif ~isfield(prior,'m_orig')  % Untransformed
    prior.m_orig = prior.m;
    prior.m = vbtransform(prior.m,LB,UB,'dir');    
end
if ~isfield(prior,'M') || isempty(prior.M)
    % prior.M = diag(width.^2);  % M = inv(W)
    prior.M = prior.nu*diag(var(X,[],2));  % M = inv(W)
    % prior.M = prior.nu*cov(X');  % M = inv(W)
    prior.M_orig = prior.nu*diag(prior.var_orig);
elseif ~isfield(prior,'M_orig') % Untransformed
    prior.M_orig = prior.M;
    if any(isfinite(LB)) || any(isfinite(UB))
        y = [];
        % Compute scale matrix in transformed space via Monte Carlo
        while size(y,1) < 1e5
            x = mvnrnd(repmat(prior.m_orig',[1e5,1]),prior.M);
            f = any(bsxfun(@le, x, LB') | bsxfun(@ge, x, UB'),2);
            x(f,:) = [];
            y = [y; vbtransform(x',LB,UB,'dir')'];
        end

        if isdiag(prior.M_orig)
            prior.M = diag(var(y));
        else
            prior.M = cov(y);        
        end
    end
end

prior.logW = -2*sum(log(diag(chol(prior.M))));

vbmodel.prior = prior;    
    
end
 
%--------------------------------------------------------------------------
function vbmodel = vbprune(vbmodel,thresh)
%VBPRUNE Remove near-empty components
if thresh <= 0; return; end

nk = sum(vbmodel.R,1);
idx = find(nk < thresh);       % Empty components
if isempty(idx); return; end
for f = {'R','alpha','beta','m','nu','logW'}
    vbmodel.(f{:})(:,idx) = [];
end
vbmodel.U(:,:,idx) = [];
vbmodel.R = bsxfun(@rdivide,vbmodel.R,sum(vbmodel.R,2));
vbmodel.logR = log(vbmodel.R);

end