function elbo_gradtest(theta,lambda,hyp,X,y)
% ELBO_GRADTEST Test gradients for ELBO combutation

if nargin < 1; theta = []; end
if nargin < 2; lambda = []; end
if nargin < 3; hyp = []; end
if nargin < 4; X = []; end
if nargin < 5; y = []; end

check_kl = 0;
check_entropy = 1;
check_quadrature = 0;
check_logjointgrad = 0;
check_gp = 0;

meanfun = 4;
quadratic_mean = meanfun == 4;


if isscalar(theta)
    D = theta;
    theta = [];
elseif isempty(theta) && isempty(X)
    D = 3;
else
    D = size(X,2);
end

if isempty(theta)
    K = 5;  % Default number of components, if not specified otherwise
    theta = [randn(D*K,1); randn(K,1)];
else
    K = numel(theta)/(D+1);     % Number of components
end

% Add LAMBDA to variational parameter vector
if isempty(lambda)
    lambda = exp(0.1*randn(D,1));
end
theta = [theta; log(lambda(:))];

if isempty(hyp)
    Ns = randi(3);    % Also test multiple hyperparameters
    hyp = [log(lambda)*ones(1,Ns); 0.1*randn(1,Ns); 0.1*randn(1,Ns)-3; 10*randn(1,Ns)];
    if quadratic_mean
        hyp = [hyp; randn(D,Ns); 0.2*randn(D,Ns)-3];
        % hyp = [hyp; theta(1:D*K); theta(D*K+1)+theta(K+D*K+(1:D))];
    end
end

% Extract variational parameters from THETA
mu(:,:) = reshape(theta(1:D*K),[D,K]);
sigma(1,:) = exp(theta(D*K+1:D*K+K));
lambda(:,1) = exp(theta(D*K+K+1:end));

if isempty(X)
    N = 100;
    X = randn(N,D);
end
if isempty(y)   % Ideally should generate from the GP but this is fine for now
    sf2 = mean(exp(hyp(D+2,:)),2);
    y = sqrt(sf2)*randn(N,1);
end

[N,D] = size(X);            % Number of training points and dimension
[Nhyp,Ns] = size(hyp);      % Hyperparameters and samples

% Compute GP posterior
gp = gplite_post(hyp,X,y,meanfun);

% Variational posterior
vp.D = D;
vp.K = K;
vp.mu = mu;
vp.sigma = sigma;
vp.lambda = lambda;
vp.LB_orig = [];
vp.UB_orig = [];
vp.trinfo = [];

if check_kl
    fprintf('---------------------------------------------------------------------------------\n');
    fprintf('Check KL-divergence computation...\n\n');

    vp2 = vp;
    vp2.mu = vp2.mu + randn(size(mu));
    vp2.sigma = vp2.sigma.*exp(randn(size(sigma)));
    vp2.lambda = vp2.lambda.*exp(randn(size(lambda)));
    kldivs = vbmc_kldiv(vp,vp2)
end

if check_entropy
    fprintf('---------------------------------------------------------------------------------\n');
    fprintf('Check approximate entropy gradient...\n\n');
    f = @(x) enttest(x,vp,0);
    derivcheck(f,theta .* (0.5 + rand(size(theta))));

    fprintf('---------------------------------------------------------------------------------\n');
    fprintf('Check Monte Carlo entropy gradient...\n\n');

    fprintf('Test only with one component first (signed errors):\n\n');
    vp1 = vp;
    idx = randi(vp.K);
    vp1.K = 1;
    vp1.mu = vp.mu(:,idx);
    vp1.sigma = vp.sigma(idx);
    theta1 = [vp1.mu(:); log(vp1.sigma(:)); log(vp1.lambda(:))];
    f = @(x) enttest(x,vp1,1e6);

    % Entropy and entropy gradient of a multivariate normal
    H_true = 0.5*D*log(2*pi*exp(1)) + D*log(vp1.sigma(1)) + sum(log(vp1.lambda));
    dH_true = [zeros(D,1); D; ones(D,1)];

    [H,dH] = f(theta1);

    [H_true-H]
    [dH_true-dH]'

    fprintf('Test with multiple (equal) components (signed errors):\n\n');
    vpm = vp;
    vpm.mu = repmat(vp.mu(:,idx),[1,K]);
    vpm.sigma = vp.sigma(idx)*ones(1,K);
    thetam = [vpm.mu(:); log(vpm.sigma(:)); log(vpm.lambda(:))];
    f = @(x) enttest(x,vpm,1e5);

    % Entropy is the same, gradient should be similar
    H_true = 0.5*D*log(2*pi*exp(1)) + D*log(vp1.sigma(1)) + sum(log(vp1.lambda));
    dH_true = [zeros(D*K,1); D*ones(K,1)/K; ones(D,1)];

    [H,dH] = f(thetam);

    [H_true-H]
    [dH_true-dH]'

    % derivcheck(f,theta .* (0.5 + rand(size(theta))),1);
end

if check_quadrature
    fprintf('---------------------------------------------------------------------------------\n');
    fprintf('Check Bayesian quadrature of log joint...\n\n');

    if quadratic_mean

        % Variational posterior and hyperparameters for D=1 and K=1
        vp1.D = 1;
        vp1.K = 1;
        vp1.mu = mu(1,1);
        vp1.sigma = sigma(1);
        vp1.lambda = lambda(1);
        vp1.LB_orig = [];
        vp1.UB_orig = [];
        hyp1 = hyp([1,vp1.D+(1:3)],1);
        if quadratic_mean; hyp1 = [hyp1; hyp([vp1.D+4;2*vp1.D+4])]; end

        % Analytical computation of integral
        ell = exp(hyp1(1));
        sf2 = exp(2*hyp1(2));
        sn2 = exp(2*hyp1(3));
        m0 = hyp1(4);
        xm = hyp1(5);
        omega = exp(hyp1(6));

        mu1 = vp1.mu;
        sigma1 = vp1.sigma;
        lambda1 = vp1.lambda;
        x1 = X(1);
        y1 = y(1);
        tau = sqrt(sigma1^2*lambda1^2 + ell^2);

        Kmat = sf2/sqrt(2*pi*ell^2) + sn2;
        F = sf2/Kmat*normpdf(x1,mu1,tau)*(y1 - m0 + 1/(2*omega^2)*(x1 - xm)^2) ...
            + m0 - 1/(2*omega^2)*(mu1^2 + sigma1^2*lambda1^2 - 2*xm*mu1 + xm^2);

        gp1 = gplite_post(hyp1,x1,y1,meanfun);

        F_ana = gplogjoint(vp1,gp1);
        F_num = gplogjoint_num(vp1,gp1);

        if D == 1
            xs = linspace(min(X)-3*std(X),max(X)+3*std(X))';
            ys = gplite_pred(gp1,xs);
            plot(xs,ys,'k-'); hold on;
            scatter(X,y,'or');
        end

        [F,F_ana, F_num]
    end

    F_ana = gplogjoint(vp,gp);
    F_num = gplogjoint_num(vp,gp);

    [F_ana, F_num]
    (F_ana - F_num) / F_ana
end

if check_logjointgrad
    fprintf('---------------------------------------------------------------------------------\n');
    fprintf('Check log joint gradient (%d samples)...\n\n',Ns);
    f = @(x) logjointtest(x,vp,gp);
    theta0 = theta .* (0.5 + rand(size(theta)));
    
    derivcheck(f,theta0',1);
    derivcheck(f,theta0);   % Sometimes this goes crazy

    fprintf('---------------------------------------------------------------------------------\n');
    fprintf('Check log joint (diagonal) variance gradient (%d samples)...\n\n',Ns);
    f = @(x) logjointvartest(x,vp,gp);
    derivcheck(f,theta .* (0.5 + rand(size(theta))));
end

%% Check GP gradients
if check_gp
    gplite_test(hyp(:,1).*exp(0.1*rand(size(hyp(:,1)))),X,y,meanfun);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [H,dH] = enttest(theta,vp,Nent)
%ENTTEST Test entropy estimation

% Extract variational parameters from THETA
D = vp.D;
K = vp.K;
vp.mu(:,:) = reshape(theta(1:D*K),[D,K]);
vp.sigma(1,:) = exp(theta(D*K+1:D*K+K));
if numel(theta) == D*K+K+D; vp.lambda(:,1) = exp(theta(D*K+K+1:end)); end

if Nent == 0
    [H,dH] = vbmc_ent(vp,[1 1 1]);
else
    [H,dH] = vbmc_entmc(vp,Nent,[1 1 1]);    
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [G,dG] = logjointtest(theta,vp,gp)
%LOGJOINTTEST Test log joint

% Extract variational parameters from THETA
D = vp.D;
K = vp.K;
vp.mu(:,:) = reshape(theta(1:D*K),[D,K]);
vp.sigma(1,:) = exp(theta(D*K+1:D*K+K));
if numel(theta) == D*K+K+D; vp.lambda(:,1) = exp(theta(D*K+K+1:end)); end

[G,dG] = gplogjoint(vp,gp);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [V,dV] = logjointvartest(theta,vp,gp)
%LOGJOINTVARTEST Test log joint variance

% Extract variational parameters from THETA
D = vp.D;
K = vp.K;
vp.mu(:,:) = reshape(theta(1:D*K),[D,K]);
vp.sigma(1,:) = exp(theta(D*K+1:D*K+K));
if numel(theta) == D*K+K+D; vp.lambda(:,1) = exp(theta(D*K+K+1:end)); end

[~,~,V,dV] = gplogjoint(vp,gp,[1 1 1],1,1,2);

end
