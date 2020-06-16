function [vp,hyp_warped] = warp_gpandvp_vbmc(trinfo,vp_old,gp_old)
%WARP_GPANDVP_VBMC Update GP hyps and variational posterior after warping.

D = size(gp_old.X,2);
trinfo_old = vp_old.trinfo;

%% Update GP hyperparameters

warpfun = @(x) warpvars_vbmc(warpvars_vbmc(x,'i',trinfo_old),'d',trinfo);

Ncov = gp_old.Ncov;
Nnoise = gp_old.Nnoise;
Nmean = gp_old.Nmean;
if ~isempty(gp_old.outwarpfun); Noutwarp = gp_old.Noutwarp; else; Noutwarp = 0; end

Ns_gp = numel(gp_old.post);
hyp_warped = NaN(Ncov+Nnoise+Nmean+Noutwarp,Ns_gp);

for s = 1:Ns_gp
    hyp = gp_old.post(s).hyp;
    hyp_warped(:,s) = hyp;
    
    % Update GP input length scales
    ell = exp(hyp(1:D))';
    [~,ell_new] = unscent_warp(warpfun,gp_old.X,ell);
    hyp_warped(1:D,s) = mean(log(ell_new),1);    % Geometric mean of length scales

    % We assume relatively no change to GP output and noise scales
    
    switch gp_old.meanfun
        case 0
            % Warp constant mean
            m0 = hyp(Ncov+Nnoise+1);
            dy_old = warpvars_vbmc(gp_old.X,'logp',trinfo_old);
            dy = warpvars_vbmc(warpfun(gp_old.X),'logp',trinfo);            
            m0w = m0 + mean(dy) - mean(dy_old);
            
            hyp_warped(Ncov+Nnoise+1,s) = m0w;
        
        case 4
            % Warp quadratic mean
            m0 = hyp(Ncov+Nnoise+1);
            xm = hyp(Ncov+Nnoise+1+(1:D))';
            omega = exp(hyp(Ncov+Nnoise+1+D+(1:D)))';
            
            % Warp location and scale
            [xmw,omegaw] = unscent_warp(warpfun,xm,omega);
                        
            % Warp maximum
            dy_old = warpvars_vbmc(xm,'logpdf',trinfo_old)';
            dy = warpvars_vbmc(xmw,'logpdf',trinfo)';
            m0w = m0 + dy - dy_old;
            
            hyp_warped(Ncov+Nnoise+1,s) = m0w;
            hyp_warped(Ncov+Nnoise+1+(1:D),s) = xmw';
            hyp_warped(Ncov+Nnoise+1+D+(1:D),s) = log(omegaw)';            
            
        otherwise
            error('Unsupported GP mean function for input warping.');
    end
end

%% Update variational posterior

vp = vp_old;
vp.trinfo = trinfo;

mu = vp_old.mu';
sigmalambda = bsxfun(@times,vp_old.lambda,vp_old.sigma)';

[muw,sigmalambdaw] = unscent_warp(warpfun,mu,sigmalambda);

vp.mu = muw';
lambdaw = sqrt(D*mean(bsxfun(@rdivide,sigmalambdaw.^2,sum(sigmalambdaw.^2,2)),1));
vp.lambda(:,1) = lambdaw(:);

sigmaw = exp(mean(log(bsxfun(@rdivide,sigmalambdaw,lambdaw)),2));
vp.sigma(1,:) = sigmaw;

% Approximate change in weight
dy_old = warpvars_vbmc(mu,'logpdf',trinfo_old)';
dy = warpvars_vbmc(muw,'logpdf',trinfo)';

ww = vp_old.w .* exp(dy - dy_old);
vp.w = ww ./ sum(ww);

end