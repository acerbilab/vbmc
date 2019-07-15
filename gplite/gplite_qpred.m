function y = gplite_qpred(gp,p,type,Xstar,ystar,s2star)
%GPLITE_QPRED Quantile prediction for lite Gaussian Processes regression.

if nargin < 5; ystar = []; end
if nargin < 6; s2star = []; end

Ns = numel(gp.post);           % Hyperparameter samples
Nstar = size(Xstar,1);         % Number of test inputs

nx = 10; 
xx = norminv(linspace(0.5/nx,1-0.5/nx,nx));

switch lower(type(1))
    case 'y'; obs_flag = true;
    case 'f'; obs_flag = false;
    otherwise
        error('gplite_qpred:unknowntype', ...
            'Quantile prediction TYPE should be ''y'' for predicted observations or ''F'' for predicted latent function.');
end

% Output warping function
outwarp_flag = isfield(gp,'outwarpfun') && ~isempty(gp.outwarpfun);
if outwarp_flag
    Noutwarp = gp.Noutwarp;
    fmu_prewarp = zeros(Nstar,Ns);
else
    Noutwarp = 0;
end

% Get GP prediction (observed or latent), by hyperparameter sample
if obs_flag
    [gmu,gs2] = gplite_pred(gp,Xstar,ystar,s2star,1,1);   
else
    [~,~,gmu,gs2] = gplite_pred(gp,Xstar,ystar,s2star,1,1);
end

y = zeros(Nstar,Ns*nx);

for s = 1:Ns
    grid = bsxfun(@plus,gmu(:,s),bsxfun(@times,sqrt(gs2(:,s)),xx));
    if outwarp_flag
        hyp = gp.post(s).hyp;
        hyp_outwarp = hyp(gp.Ncov+gp.Nnoise+gp.Nmean+1:gp.Ncov+gp.Nnoise+gp.Nmean+Noutwarp);
        grid = gp.outwarpfun(hyp_outwarp,grid,'inv');
    end
    y(:,(1:nx)+(s-1)*nx) = grid;
end

y = quantile(y,p,2);

