function acq = acqimiqr_vbmc(Xs,vp,gp,optimState,fmu,fs2,fbar,vtot)
%VBMC_ACQIMIQR Integrated median interquantile range acquisition function.

u = 0.6745; % norminv(0.75)

if isempty(Xs)
    % Return acquisition function info struct
    acq.importance_sampling = true;
    acq.importance_sampling_vp = false;
    acq.log_flag = true;
    return;
elseif ischar(Xs)
    switch lower(Xs)
        case 'islogf1'
        % Importance sampling log base proposal (shared part)
            acq = fmu;
        case 'islogf2'
        % Importance sampling log base proposal (added part)
        % (Full log base proposal is fixed + added)
            fs = sqrt(fs2);
            acq = u*fs + log1p(-exp(-2*u*fs));
        case 'islogf'
        % Importance sampling log base proposal distribution
            fs = sqrt(fs2);
            acq = fmu + u*fs + log1p(-exp(-2*u*fs));
    end
    return;
end

% Different importance sampling inputs for different GP hyperparameters?
multipleinputs_flag = size(optimState.ActiveImportanceSampling.Xa,3) > 1;

% Xs is in *transformed* coordinates

[Nx,D] = size(Xs);
Ns = size(fmu,2);
Na = size(optimState.ActiveImportanceSampling.Xa,1);

% Estimate observation noise at test points from nearest neighbor
[~,pos] = min(sq_dist(bsxfun(@rdivide,Xs,optimState.gplengthscale),gp.X_rescaled),[],2);
sn2 = gp.sn2new(pos);
% sn2 = min(sn2,1e4);
ys2 = fs2 + sn2;    % Predictive variance at test points

if multipleinputs_flag
    Xa = zeros(Na,D);
else
    Xa = optimState.ActiveImportanceSampling.Xa;
end
acq = zeros(Nx,Ns);

%% Compute integrated acquisition function via importance sampling

for s = 1:Ns    
    hyp = gp.post(s).hyp;
    L = gp.post(s).L;
    Lchol = gp.post(s).Lchol;
    sn2_eff = 1/gp.post(s).sW(1)^2;
    
    if multipleinputs_flag
        Xa(:,:) = optimState.ActiveImportanceSampling.Xa(:,:,s);
    end
    
    % Compute cross-kernel matrix Ks_mat
    if gp.covfun(1) == 1    % Hard-coded SE-ard for speed
        ell = exp(hyp(1:D))';
        sf2 = exp(2*hyp(D+1));
        Ks_mat = sq_dist(gp.X*diag(1./ell),Xs*diag(1./ell));
        Ks_mat = sf2 * exp(-Ks_mat/2);
        
        Ka_mat = sq_dist(Xa*diag(1./ell),Xs*diag(1./ell));
        Ka_mat = sf2 * exp(-Ka_mat/2);
        
        %Kax_mat = sq_dist(Xa*diag(1./ell),gp.X*diag(1./ell));
        %Kax_mat = sf2 * exp(-Kax_mat/2);
        Kax_mat(:,:) = optimState.ActiveImportanceSampling.Kax_mat(:,:,s);
    else
        error('Other covariance functions not supported yet.');
    end
    
    if Lchol
        C = Ka_mat' - Ks_mat'*(L\(L'\Kax_mat'))/sn2_eff;
    else
        C = Ka_mat' + Ks_mat'*(L*Kax_mat');        
    end
            
    tau2 = bsxfun(@rdivide,C.^2,ys2(:,s));
    s_pred = sqrt(max(bsxfun(@minus,optimState.ActiveImportanceSampling.fs2a(:,s)',tau2),0));
    
    lnw = optimState.ActiveImportanceSampling.lnw(s,:);
    
    zz = bsxfun(@plus,lnw,u*s_pred + log1p(-exp(-2*u*s_pred)));
    lnmax = max(zz,[],2);
    acq(:,s) = log(sum(exp(bsxfun(@minus,zz,lnmax)),2)) + lnmax;    
end

if Ns > 1
    M = max(acq,[],2);
    acq = M + log(sum(exp(bsxfun(@minus,acq,M)),2)/Ns);    
end

end


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