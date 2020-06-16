function ActiveImportanceSampling = activeimportancesampling_vbmc(vp,gp,acqfun,acqinfo,options)
%ACTIVEIMPORTANCESAMPLING_VBMC Setup importance sampling acquisition functions

% This function samples from the base importance sampling (IS) density in
% three steps:
% 1) Use importance sampling-resampling (ISR) to sample from the
%    base IS density based on a proposal distribuion which is a mixture of
%    a smoothed variational posterior and on box-uniform mixture centered 
%    around the current training points
% 2) Optionally use MCMC initialized with the ISR samples to sample from 
%    the base IS density
% 3) Compute IS statistics and book-keeping

% Does the importance sampling step use the variational posterior?
isamplevp_flag = isfield(acqinfo,'importance_sampling_vp') ...
    && acqinfo.importance_sampling_vp;

% Do we simply sample from the variational posterior?
onlyvp_flag = isfield(acqinfo,'variational_importance_sampling') ...
    && acqinfo.variational_importance_sampling;

D = size(gp.X,2);
Ns_gp = numel(gp.post); % # GP hyperparameter samples

% Input space bounds and typical scales (for MCMC only)
widths = std(gp.X,[],1);
MaxBnd = 0.5;
diam = max(gp.X) - min(gp.X);
LB = min(gp.X) - MaxBnd*diam;
UB = max(gp.X) + MaxBnd*diam;

ActiveImportanceSampling.lnw = [];
ActiveImportanceSampling.Xa = [];
ActiveImportanceSampling.fs2a = [];

if onlyvp_flag
    %% Step 0: Simply sample from the variational posterior
    
    Na = options.ActiveImportanceSamplingMCMCSamples;
    Xa = vbmc_rnd(vp,Na,0);
    [~,~,fmu,fs2] = gplite_pred(gp,Xa,[],[],1,0);        
        
    if isfield(acqinfo,'mcmc_importance_sampling') && acqinfo.mcmc_importance_sampling
        
        % Compute fractional effective sample size (ESS)
        fESS = fess_vbmc(vp,fmu,Xa);
        
        % If fESS is less than thresh ==> major mismatch, do MCMC
        if fESS < options.ActiveImportanceSamplingfESSThresh
            Xa_old = Xa;        
            
            if isamplevp_flag
                logpfun = @(x) log_isbasefun(x,acqfun,gp,vp);
            else
                logpfun = @(x) log_isbasefun(x,acqfun,gp,[]);
            end
            
            % Get MCMC options
            Nmcmc_samples = Na*options.ActiveImportanceSamplingMCMCThin;
            thin = 1;
            burnin = 0;
            sampleopts = get_mcmcopts([],thin,burnin);
            logPfuns = logpfun;
            W = Na;    % # walkers

            % Perform a single MCMC step for all samples
            Xa = eissample_lite(logPfuns,Xa,Nmcmc_samples,W,widths,LB,UB,sampleopts);
            Xa = Xa(end-Na+1:end,:);
            [~,~,fmu,fs2] = gplite_pred(gp,Xa,[],[],1,0);            
            
            if 0
                hold off;
                scatter(Xa_old(:,1),Xa_old(:,2),'b'); hold on;
                scatter(Xa(:,1),Xa(:,2),'k'); hold on;
                drawnow;
            end
        end
    end
        
    if isamplevp_flag
        vlnpdf = max(vbmc_pdf(vp,Xa,0,1),log(realmin));
        lny = acqfun('islogf1',vlnpdf,[],[],fmu,fs2);
    else
        lny = acqfun('islogf1',[],[],[],fmu,fs2);
    end

    ActiveImportanceSampling.fs2a = fs2;
    ActiveImportanceSampling.lnw = lny';
    ActiveImportanceSampling.Xa = Xa;
    
else
    %% Step 1: Importance sampling-resampling

    Nvp_samples = options.ActiveImportanceSamplingVPSamples;
    Nbox_samples = options.ActiveImportanceSamplingBoxSamples;
    w_vp = Nvp_samples/(Nvp_samples + Nbox_samples);

    rect_delta = 2*std(gp.X);

    % Smoothed posterior for importance sampling-resampling
    if Nvp_samples > 0                
        scale_vec = [0.05,0.2,1];
        %scale = sqrt(mean(var(gp.X,[],1)));

        vp_is = vp;
        for ii = 1:numel(scale_vec)
            vp_is.K = vp_is.K + vp.K;
            vp_is.w = [vp_is.w, vp.w];
            vp_is.mu = [vp_is.mu, vp.mu];
            vp_is.sigma = [vp_is.sigma, sqrt(vp.sigma.^2 + scale_vec(ii)^2)];
        end
        vp_is.w = vp_is.w/sum(vp_is.w);

        % Sample from smoothed posterior
        Xa_vp = vbmc_rnd(vp_is,Nvp_samples,0,0);
        [lnw,fs2a_vp] = activesample_proposalpdf(Xa_vp,gp,vp_is,w_vp,rect_delta,acqfun,vp,isamplevp_flag);
        ActiveImportanceSampling.lnw = [ActiveImportanceSampling.lnw, lnw'];
        ActiveImportanceSampling.Xa = [ActiveImportanceSampling.Xa; Xa_vp];
        ActiveImportanceSampling.fs2a = [ActiveImportanceSampling.fs2a; fs2a_vp];
    else
        vp_is = [];
    end

    % Box-uniform sampling around training inputs
    if Nbox_samples > 0
        jj = randi(size(gp.X,1),[1,Nbox_samples]);
        Xa_box = gp.X(jj,:) + bsxfun(@times,2*rand(numel(jj),D)-1,rect_delta);
        [lnw,fs2a_box] = activesample_proposalpdf(Xa_box,gp,vp_is,w_vp,rect_delta,acqfun,vp,isamplevp_flag);
        ActiveImportanceSampling.lnw = [ActiveImportanceSampling.lnw, lnw'];
        ActiveImportanceSampling.Xa = [ActiveImportanceSampling.Xa; Xa_box];
        ActiveImportanceSampling.fs2a = [ActiveImportanceSampling.fs2a; fs2a_box];
    end

    ActiveImportanceSampling.lnw(~isfinite(ActiveImportanceSampling.lnw)) = -Inf;
    % optimState.w = exp(optimState.lnw - max(optimState.lnw))';
    % optimState.w = optimState.w / sum(optimState.w);
    % 1./sum(optimState.w.^2)

    %% Step 2 (optional): MCMC sample

    Nmcmc_samples = options.ActiveImportanceSamplingMCMCSamples;
    
    if Nmcmc_samples > 0
        
        ActiveImportanceSampling_old = ActiveImportanceSampling; 
        
        ActiveImportanceSampling.lnw = zeros(Ns_gp,Nmcmc_samples);
        ActiveImportanceSampling.Xa = zeros(Nmcmc_samples,D,Ns_gp);
        ActiveImportanceSampling.fs2a = zeros(Nmcmc_samples,Ns_gp);        
        

        gp1 = gp;   % Consider only one GP sample at a time
        
        for s = 1:Ns_gp
            
            gp1.post = [];  % Assign current GP sample
            gp1.post = gp.post(s);
            
            if D == 2 && 0
                % We could use a quasi-random grid for D <= 2, but not implemented
                Xa = rand(Na,D).*(UB - LB) + LB;
                [~,~,fmu,optimState.fs2a] = gplite_pred(gp1,Xa,[],[],1,0);                    
                optimState.lnw = mean(fmu,2)';
                optimState.Xa = Xa;                    
            else
                if isamplevp_flag
                    logpfun = @(x) log_isbasefun(x,acqfun,gp1,vp);
                else
                    logpfun = @(x) log_isbasefun(x,acqfun,gp1,[]);
                end

                % Get MCMC options
                thin = options.ActiveImportanceSamplingMCMCThin;
                burnin = ceil(thin*Nmcmc_samples/2);
                sampleopts = get_mcmcopts(Nmcmc_samples,thin,burnin);
                
                logPfuns = logpfun;
                % sampleopts.TransitionOperators = {'transSliceSampleRD'};
                W = 2*(D+1);    % # walkers

                if 0
                    % Take starting points from high posterior density region
                    hpd_frac = 0.5;
                    N = numel(gp1.y);
                    N_hpd = min(N,max(W,round(hpd_frac*N)));
                    [~,ord] = sort(gp1.y,'descend');
                    X_hpd = gp1.X(ord(1:N_hpd),:);
                    x0 = X_hpd(randperm(N_hpd,min(W,N_hpd)),:);
                    x0 = bsxfun(@min,bsxfun(@max,x0,LB),UB);
                else
                    % Use importance sampling-resampling
                    [~,~,fmu,fs2] = gplite_pred(gp1,ActiveImportanceSampling_old.Xa,[],[],1,0);               
                    lnw = ActiveImportanceSampling_old.lnw(s,:) + acqfun('islogf2',[],[],[],fmu,fs2)';
                    w = exp(bsxfun(@minus,lnw,max(lnw,[],2)));
                    x0 = zeros(W,D);
                    for ii = 1:W
                        idx = catrnd(w,1);
                        w(idx) = 0;
                        x0(ii,:) = ActiveImportanceSampling_old.Xa(idx,:);
                    end
                end

                [Xa,logp] = eissample_lite(logPfuns,x0,Nmcmc_samples,W,widths,LB,UB,sampleopts);
                [~,~,fmu,fs2] = gplite_pred(gp1,Xa,[],[],1,0);               

                % Fixed log weight for importance sampling (log fixed integrand)
                if isamplevp_flag
                    vlnpdf = max(vbmc_pdf(vp,Xa,0,1),log(realmin));            
                    lny = acqfun('islogf1',vlnpdf,[],[],fmu,fs2);
                else
                    lny = acqfun('islogf1',[],[],[],fmu,fs2);
                end
                % lny = lny - warpvars_vbmc(Xa,'logp',vp.trinfo);

                ActiveImportanceSampling.fs2a(:,s) = fs2;
                ActiveImportanceSampling.lnw(s,:) = bsxfun(@minus,lny',logp');
                ActiveImportanceSampling.Xa(:,:,s) = Xa;
            end

        end
    end
end

if 0
    hold off;
    scatter(ActiveImportanceSampling.Xa(:,1),ActiveImportanceSampling.Xa(:,2)); hold on;
    % scatter(x0(:,1),x0(:,2),'ro','MarkerFaceColor','r')
    xlim([-2,2]);
    ylim([-2,2]);
    drawnow;
end

%% Step 3: Pre-compute quantities for importance sampling calculations

% Precompute cross-kernel matrix on importance points
Kax_mat = zeros(size(ActiveImportanceSampling.Xa,1),size(gp.X,1),Ns_gp);
for s = 1:Ns_gp
    if size(ActiveImportanceSampling.Xa,3) == 1
        Xa = ActiveImportanceSampling.Xa;
    else
        Xa(:,:) = ActiveImportanceSampling.Xa(:,:,s);
    end
    hyp = gp.post(s).hyp;
    if gp.covfun(1) == 1    % Hard-coded SE-ard for speed
        ell = exp(hyp(1:D))';
        sf2 = exp(2*hyp(D+1));        
        Kax_tmp = sq_dist(Xa*diag(1./ell),gp.X*diag(1./ell));
        Kax_mat(:,:,s) = sf2 * exp(-Kax_tmp/2);        
    else
        error('Other covariance functions not supported yet.');
    end
end
ActiveImportanceSampling.Kax_mat = Kax_mat;

% Precompute integrated mean basis function on importance points
if isfield(gp,'intmeanfun') && gp.intmeanfun > 0
    plus_idx = gp.intmeanfun_var > 0;    
    if size(ActiveImportanceSampling.Xa,3) == 1
        Ha = gplite_intmeanfun(ActiveImportanceSampling.Xa,gp.intmeanfun);
        ActiveImportanceSampling.Ha = Ha(plus_idx,:);
    else
        for s = 1:Ns_gp
            Ha = gplite_intmeanfun(ActiveImportanceSampling.Xa(:,:,s),gp.intmeanfun);
            ActiveImportanceSampling.Ha(:,:,s) = Ha(plus_idx,:);
        end
    end
end


end

%--------------------------------------------------------------------------
function [lnw,fs2] = activesample_proposalpdf(Xa,gp,vp_is,w_vp,rect_delta,acqfun,vp,isamplevp_flag)
%ACTIVESAMPLE_PROPOSALPDF Compute importance weights for proposal pdf

[N,D] = size(gp.X);
Na = size(Xa,1);

[~,~,fmu,fs2] = gplite_pred(gp,Xa,[],[],1,0);

Ntot = 1 + N; % Total number of mixture elements

if w_vp < 1; templpdf = zeros(Na,Ntot); end

% Mixture of variational posteriors
if w_vp > 0
    logflag = true;
    templpdf(:,1) = vbmc_pdf(vp_is,Xa,0,logflag) + log(w_vp);
else
    templpdf(:,1) = -Inf;
end

% Fixed log weight for importance sampling (log fixed integrand)
if isamplevp_flag
    vlnpdf = max(vbmc_pdf(vp,Xa,0,1),log(realmin));
    lny = acqfun('islogf1',vlnpdf,[],[],fmu,fs2);
else
    lny = acqfun('islogf1',[],[],[],fmu,fs2);
end
% lny = lny - warpvars_vbmc(Xa,'logp',vp.trinfo);

% Mixture of box-uniforms
if w_vp < 1
    VV = prod(2*rect_delta);

    for ii = 1:N
        templpdf(:,ii+1) = log(all(abs(bsxfun(@minus,Xa,gp.X(ii,:))) < rect_delta,2) / VV / N * (1 - w_vp));
    end

    mmax = max(templpdf,[],2);
    lpdf = log(sum(exp(templpdf - mmax),2));
    lnw = bsxfun(@minus,lny,lpdf + mmax);
else
    lnw = bsxfun(@minus,lny,templpdf);
end

end

%--------------------------------------------------------------------------
function y = log_isbasefun(x,acqfun,gp,vp)
%LOG_ISBASEFUN Base importance sampling proposal log pdf

[fmu,fs2] = gplite_pred(gp,x);
if isempty(vp)
    y = acqfun('islogf',[],[],[],fmu,fs2);
else
    vlnpdf = max(vbmc_pdf(vp,x,0,1),log(realmin));
    y = acqfun('islogf',vlnpdf,[],[],fmu,fs2);
end

end


%--------------------------------------------------------------------------
%SQ_DIST Compute matrix of all pairwise squared distances between two sets 
% of vectors, stored in the columns of the two matrices, a (of size n-by-D) 
% and b (of size m-by-D).
function C = sq_dist(a,b)

n = size(a,1);
m = size(b,1);
mu = (m/(n+m))*mean(b,1) + (n/(n+m))*mean(a,1);
a = bsxfun(@minus,a,mu); b = bsxfun(@minus,b,mu);
C = bsxfun(@plus,sum(a.*a,2),bsxfun(@minus,sum(b.*b,2)',2*a*b'));
C = max(C,0);

end

%--------------------------------------------------------------------------
function sampleopts = get_mcmcopts(Ns,thin,burnin)
%GET_MCMCOPTS Get standard MCMC options.

if nargin < 2 || isempty(thin); thin = 1; end
if nargin < 3; burnin = []; end

sampleopts.Thin = thin;
if isempty(burnin)
    sampleopts.Burnin = ceil(sampleopts.Thin*Ns/2);
else
    sampleopts.Burnin = burnin;
end
sampleopts.Display = 'off';
sampleopts.Diagnostics = false;
sampleopts.VarTransform = false;
sampleopts.InversionSample = false;
sampleopts.FitGMM = false;

end

%--------------------------------------------------------------------------
function x = catrnd(p,n)
%CATRND Sample from categorical distribution.

maxel = 1e6;
Nel = n*numel(p);
stride = ceil(maxel/numel(p));

cdf(1,:) = cumsum(p);
u = rand(n,1)*cdf(end);

% Split for memory reasons
if Nel <= maxel
    x = sum(bsxfun(@lt, cdf, u),2) + 1;
else
    x = zeros(n,1);
    idx_min = 1;
    while idx_min <= n
        idx_max = min(idx_min+stride-1,n);
        idx = idx_min:idx_max;
        x(idx) = sum(bsxfun(@lt, cdf, u(idx)),2) + 1;
        idx_min = idx_max+1;
    end
end

end
