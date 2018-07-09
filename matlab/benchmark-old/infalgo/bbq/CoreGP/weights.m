function [w]=weights(gp,KSinv_NS_KSinv)

NSamples=numel(gp.hypersamples);
if nargin<2
    KSinv_NS_KSinv=gp.KSinv_NS_KSinv;
end

if isfield(gp, 'grad_hyperparams')
    derivs = gp.grad_hyperparams;
elseif isfield(gp, 'use_derivatives') 
    derivs = gp.use_derivatives;
elseif isfield(gp, 'derivs_cov') 
    derivs = gp.derivs_cov;
elseif isfield(gp, 'derivs_mean') 
    derivs = gp.derivs_mean;
elseif ~isfield(gp, 'use_derivatives') && isfield(gp, 'covfn') && (nargin(gp.covfn)~=1)
    derivs=true;
elseif isfield(gp.hypersamples,'glogL') && ...
        ~isempty(gp.hypersamples(1).glogL)
    derivs=true;
else
    % we can't determine the gradient of the covariance wrt hyperparams
    derivs=false;
end

if isfield(gp,'active_hp_inds')
    active_hp_inds = gp.active_hp_inds;
end



logLvec=[gp.hypersamples(:).logL];
logLvec = logLvec(:);

logLvec=(logLvec-max(logLvec)); 
Lvec=exp(logLvec);
if derivs
    [glogLcell{1:NSamples}]=gp.hypersamples(:).glogL; % actually glogl is a cell itself
    glogLmat=cell2mat(cat(2,glogLcell{:}))';
    glogLmat = glogLmat(:,active_hp_inds);
    % multiply by likelihood to get derivative of L from derivative of logL
    gLmat=fliplr(repmat(Lvec,1,size(glogLmat,2)).*glogLmat); 
    %fliplr because derivs are actually in reverse order
    
    Lvec=[Lvec;gLmat(:)];
end


if iscell(KSinv_NS_KSinv)
    w = kronmult(KSinv_NS_KSinv, Lvec);
else
    w = KSinv_NS_KSinv*Lvec;
end

w = max(w, 0);
w = w / sum(w);
