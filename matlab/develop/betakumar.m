function betakumar(theta_b)

Nx = 20;
LB = [-20 -20];
UB = [20 20];

betadata = preprocessbeta(theta_b);

fun = @(lntheta_k) betakumar_kl(lntheta_k,betadata);

% Find starting point on a grid
xxa = linspace(LB(1),UB(1),Nx);
xxb = linspace(LB(2),UB(2),Nx);
xx = [log(theta_b); combvec(xxa,xxb)'];

f = zeros(size(xx,1),1);
for i = 1:size(xx,1); f(i) = fun(xx(i,:)); end

[~,idx] = min(f);
lntheta_k0 = xx(idx,:);

optoptions = optimoptions('fmincon','GradObj','off','Display','iter');

[lntheta_k,ch] = fmincon(fun,lntheta_k0,[],[],[],[],LB,UB,[],optoptions);

Nx = 1e6;
xx = linspace(1/Nx,1-1/Nx,Nx);
alpha_k = exp(lntheta_k(1));
beta_k = exp(lntheta_k(2));

theta_b, exp(lntheta_k), ch

bpdf = betapdf(xx,theta_b(1),theta_b(2));
kpdf = alpha_k.*beta_k.*(xx.^(alpha_k-1).*(1-xx.^alpha_k).^(beta_k-1));

plot(xx,bpdf); hold on;
plot(xx,kpdf);


end

function betadata = preprocessbeta(theta_b)

Nx = 1e6;
smallthresh = eps/Nx;
largethresh = 1;

alpha_b = theta_b(1);
beta_b = theta_b(2);
betadata.theta_b = theta_b;
nf_b = exp(gammaln(alpha_b+beta_b)-gammaln(alpha_b)-gammaln(beta_b));
betadata.nf_b = nf_b;
betadata.psiab = psi(alpha_b)-psi(alpha_b+beta_b);

% First rough integration
xx0 = linspace(1/Nx,1-1/Nx,ceil(sqrt(Nx)));
bpdf0 = [0,xx0.^(alpha_b-1).*(1-xx0).^(beta_b-1)*nf_b,0];
xx0 = [0,xx0,0];

if alpha_b >= 1 && beta_b >=1
    % Only integrate for non-negligible values of the Beta density
    idx = bpdf0 > smallthresh;
    chunks = find(diff(idx));
elseif alpha_b < 1 && beta_b < 1
    % Divide in three
    idx1 = find(bpdf0(2:end-1) < largethresh,1,'first')+1;
    idx2 = find(bpdf0(2:end-1) < largethresh,1,'last')+1;
    chunks = [1 idx1-1,idx1 idx2-1,idx2,numel(bpdf0)-1];    
elseif alpha_b < 1
    
    
else
    
    
    
end

Nchunks = numel(chunks)/2;
for iChunk = 1:Nchunks
    betadata.intbounds(iChunk,:) = [xx0(chunks(iChunk*2-1)),xx0(chunks(iChunk*2)+1)];
end

end

function [f,df] = betakumar_kl(lntheta_k,betadata)

Nx = 1e6;

alpha_k = exp(lntheta_k(:,1));
beta_k = exp(lntheta_k(:,2));
alpha_b = betadata.theta_b(1);
beta_b = betadata.theta_b(2);

nf_b = betadata.nf_b;
psiab = betadata.psiab;
        
Nchunks = size(betadata.intbounds,1);
f = zeros(size(lntheta_k,1),1);
for iChunk = 1:Nchunks
    xx = linspace(betadata.intbounds(iChunk,1),betadata.intbounds(iChunk,2),Nx);
    dx = xx(2)-xx(1);

    bpdf = xx.^(alpha_b-1).*(1-xx).^(beta_b-1);
    xxa = xx.^alpha_k;
    l1xxa = log1p(-xxa);

    yy = bpdf.*(beta_k-1).*l1xxa;
    yy(:,[1 end]) = 0;

    % psiab = (psi(alpha_b)-psi(alpha_b+beta_b));
    f = f-qtrapz(yy,2)*nf_b*dx;
end

f = f - log(alpha_k.*beta_k) - (alpha_k-1)*psiab;

% Compute derivatives (doesn't seem to help much, integration is a problem)
if nargout > 1
    df = zeros(size(lntheta_k,1),2);
    
    % Derivative with respect to alpha_k
    dyy = bpdf.*(-alpha_k.*(beta_k-1).*(xx.^(alpha_k-1))./(1-xxa));
    dyy(:,[1 end]) = 0;
    df(:,1) = (-qtrapz(dyy,2)*nf_b*dx - 1./alpha_k - psiab).*alpha_k;

    % Derivative with respect to beta_k
    dyy = bpdf.*l1xxa;
    dyy(:,[1 end]) = 0;
    df(:,2) = (-qtrapz(dyy,2)*nf_b*dx - 1./beta_k).*beta_k;
end

end