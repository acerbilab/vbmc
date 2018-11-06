function gpmcdemo
%GPMCDEMO Simple demo that shows that sampling from the log mean posterior 
% GP is not the same thing as averaging sampled distribution from the 
% posterior of log GPs.

N = 5;
Nstar = 100;
Niter = 1e4;

ell = 0.1;
sf2 = 10^2;
sn2 = 1e-6;

X = rand(N,1);

% Compute kernel matrix K_mat
K_mat = sq_dist(diag(1./ell)*X');
K_mat = sf2 * exp(-K_mat/2);
L = chol(K_mat+sn2*eye(N));

% Generate observations
y = (randn(1,N)*L)';

% Compute posterior
alpha = L\(L'\y);

Xstar = linspace(0,1,Nstar)';

kss = sq_dist(diag(1./ell)*Xstar');
kss = sf2*exp(-kss/2);
Ks_mat = sq_dist(diag(1./ell)*X',diag(1./ell)*Xstar');
Ks_mat = sf2 * exp(-Ks_mat/2);

subplot(2,1,1);

fmu = Ks_mat'*alpha;
invL = L\(L'\eye(N));
fCov = kss - Ks_mat'*(invL*Ks_mat);
fCov = 0.5*(fCov + fCov');

hrnd = zeros(1,Nstar);
ystar = mvnrnd(fmu,fCov,Niter);

for i = 1:Niter
    if mod(i,100) == 0
        plot(Xstar,ystar(i,:),'-','LineWidth',0.5,'Color',0.5*[1 1 1]); hold on;
    end
    
    % Sample from pdf
    p = exp(ystar(i,:)-max(ystar(i,:)));
    p = p/sum(p);  
    hrnd = hrnd + mnrnd(100,p);    
end
Ns = sum(hrnd);
hrnd = hrnd/Ns;

plot(Xstar,fmu,'k-','LineWidth',2);
plot(X,y,'bo','MarkerFaceColor','b');

set(gcf,'Color','w');
set(gca,'TickDir','out');
xlabel('x');
ylabel('f(x)');
box off;

% Sample from mean log pdf
p = exp(fmu-max(fmu));
p = p/sum(p);
hrnd_mean = mnrnd(Ns,p)/Ns;

% Sample from mean log pdf corrected
% z = fmu + 0.5*diag(fCov);
% p = exp(z-max(z));
% p = p/sum(p);
% hrnd_means2 = mnrnd(Ns,p)/Ns;

subplot(2,1,2);

plot(Xstar,hrnd,'k:','LineWidth',2); hold on;
plot(Xstar,hrnd_mean,'k-','LineWidth',2);
% plot(Xstar,hrnd_means2,'r-','LineWidth',2);

set(gca,'TickDir','out');
xlabel('x');
ylabel('p(x)');
box off;

end