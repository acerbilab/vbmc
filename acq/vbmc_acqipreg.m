function acq = vbmc_acqipreg(Xs,vp,gp,optimState,Nacq,transpose_flag)
%VBMC_ACQIPREG Acquisition function for integrated prospective uncertainty search.

% Xs is in *transformed* coordinates

if nargin < 6 || isempty(transpose_flag); transpose_flag = false; end

% Transposed input (useful for CMAES)
if transpose_flag; Xs = Xs'; end

[N,D] = size(Xs);    % Number of points and dimension

% Threshold on GP variance, try not to go below this
TolVar = optimState.TolGPVar;

% Probability density of variational posterior at test points
p = max(vbmc_pdf(Xs,vp,0),realmin);

% GP mean and variance for each hyperparameter sample
[~,~,fmu,fs2] = gplite_pred(gp,Xs,[],1);

Ns = size(fmu,2);
fbar = sum(fmu,2)/Ns;   % Mean across samples
vbar = sum(fs2,2)/Ns;   % Average variance across samples
if Ns > 1; vf = sum((fmu - fbar).^2,2)/(Ns-1); else; vf = 0; end  % Sample variance
vtot = vf + vbar;       % Total variance

acq = zeros(N,1);
z = optimState.ymax;

% Loop over hyperparameter samples
for s = 1:Ns
    hyp = gp.post(s).hyp;

    % Extract length scales from HYP
    elltilde = exp(hyp(1:D))'/sqrt(2);
    elltilde_prod = prod(elltilde);
    sf2 = fs2(:,s) + vf;
    temp = zeros(N,1);
    
    for k = 1:vp.K    
        sigma2tilde_k = elltilde.^2 + (vp.sigma(k)*vp.lambda').^2;
        nf = elltilde_prod/sqrt(prod(sigma2tilde_k));
        
        temp = temp + ...
            vp.w(k)*nf.*exp(-0.5*sum(bsxfun(@rdivide,bsxfun(@minus,Xs,vp.mu(:,k)').^2,sigma2tilde_k),2));
    end
    
    acq = acq - sf2.*temp/Ns;
end

acq = acq .* exp(fbar-z);

% Regularization: penalize points where GP uncertainty is below threshold
idx = vtot < TolVar;
if any(idx)
    acq(idx) = acq(idx) .* exp(-(TolVar./vtot(idx)-1));
end
acq = max(acq,-realmax);

% Transposed output
if transpose_flag; acq = acq'; end

end