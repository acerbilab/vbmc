function gplite_test(hyp,X,y,meanfun)
%GPLITE_TEST Test computations for lite GP functions.

if nargin < 1; hyp = []; end
if nargin < 2; X = []; end
if nargin < 3; y = []; end
if nargin < 4 || isempty(meanfun); meanfun = 'negquad'; end

D = 2;

if isempty(hyp)
    Ns = randi(3);    % Test multiple hyperparameters
    hyp = [randn(D,Ns); 0.2*randn(1,Ns); 0.3*randn(1,Ns); 10*randn(1,Ns); randn(D,Ns); randn(D,Ns)];
end

if isempty(X)
    N = 100;
    X = randn(N,D);
end
if isempty(y)   % Ideally should generate from the GP but this is fine for now
    sf2 = mean(2*exp(hyp(D+1,:)),2);
    y = sqrt(sf2)*randn(N,1);
end

[N,D] = size(X);            % Number of training points and dimension
[Nhyp,Ns] = size(hyp);      % Hyperparameters and samples

hyp0 = hyp(:,1);

gp = gplite_post(hyp0,X,y,meanfun);

fprintf('---------------------------------------------------------------------------------\n');
fprintf('Check GP marginal likelihood computation...\n\n');
f = @(x) gplite_nlZ(x,gp);
derivcheck(f,hyp0.*exp(0.1*rand(size(hyp0))));

fprintf('---------------------------------------------------------------------------------\n');
fprintf('Check GP hyperparameters log prior computation...\n\n');
hprior.mu = randn(size(hyp0));
hprior.sigma = exp(randn(size(hyp0)));
hprior.df = exp(randn(size(hyp0))).*(randi(2,size(hyp0))-1);
f = @(x) gplite_hypprior(x,hprior);
derivcheck(f,hyp0.*exp(0.1*rand(size(hyp0))));

fprintf('---------------------------------------------------------------------------------\n');
fprintf('Check GP training with warped data...\n\n');

D = 1;
s2 = 2;
Xt = linspace(-3,3,51)';
yt = -0.5*Xt.^2/s2;
LB = 0; UB = 5;
warp.LB = LB; warp.UB = UB;
trinfo = warpvars(D,LB,UB);
trinfo.type = 9;
trinfo.alpha = 3;
trinfo.beta = 0.1;
X = warpvars(Xt,'inv',trinfo);
y = yt - warpvars(Xt,'logp',trinfo);
Ncov = D+1;
Nmean = gplite_meanfun([],X,meanfun);
Nhyp0 = Ncov+1+Nmean+2*D;

warp.LB = LB; warp.UB = UB; warp.logpdf_flag = 1;
hyp0 = zeros(Nhyp0,1);
[gp,hyp] = gplite_train(hyp0,0,X,y,meanfun,[],warp);
plot(Xt,yt); hold on;
plot(X,y); plot(gp.X,gp.y);


end