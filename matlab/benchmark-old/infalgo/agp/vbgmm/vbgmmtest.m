%VBGMMTEST Test of VBGMM capabilities.

%% 1. Bla bla

if 0
LB = [-1;-1];
UB = [1;1];
% LB = []; UB = [];
mu(:,1) = -1*[1;1];
mu(:,2) = [0;0];
sigma(:,:,1) = 0.5.^2*[1 0.8; 0.8 1];
sigma(:,:,2) = 0.2.^2*[1 0; 0 1];
n = 1e4;
k = size(mu,1);

X = [];
for i = 1:k
    X = [X; mvnrnd(ones(n,1)*mu(:,i)',sigma(:,:,i))];
end

prior = [];
prior.LB = LB;
prior.UB = UB;

options = [];
options.Nstarts = 1;
[~,vbmodel] = vbgmmfit(X',[],prior,options);

xx = vbgmmrnd(vbmodel,1e5);
vbcornerplot(X,[],[],[LB,UB]',[],vbmodel);
vbcornerplot(xx,[],[],[LB,UB]',[],vbmodel);

set(gcf,'Color','w');
set(gca,'TickDir','out');
box off;
end

%% 2 Bla bla

k = 10;
d = 5;
% nmax = 2000;
nmax = 50*d;
% LB = zeros(1,d); UB = 10*ones(1,d);
LB = []; UB = [];
Nchains = 3;

prior = []; options = [];
prior.LB = LB; prior.UB = UB;
options.Nstarts = 5;
options.Niter = 1e4;

if 0
    [X,Y,gmm] = vbgmmbenchgmm(d,k,nmax,LB,UB);
    [~,vbmodel] = vbgmmfit(X',30,prior,options);
    vbcornerplot(X,[],[],[min(X);max(X)],[],vbmodel);
end

X = []; Y = []; logZ = []; slogZ = []; lnf = []; logZu = [];
for i = 1:200
    fprintf('%d..',i);
    
    if 1
        % Generate dataset from random mixture of Gaussians
        for iChain = 1:Nchains
            [X{iChain},Y{iChain},gmm] = vbgmmbenchgmm(d,k,nmax,LB,UB);
            Y{iChain} = Y{iChain} + 100;
        end
    else
        fun = @(x) -0.5*(sqrt(sum(x.^2,2)) - 5).^2/2^2;
        % Compute normalizing constant
        rho = linspace(0,20,1e5);
        lnz = log(qtrapz(rho.^(d-1).*exp(-0.5*(rho-5).^2/2^2))*diff(rho(1:2))) + log(2*pi^(d/2)) - gammaln(d/2);
        for iChain = 1:Nchains
            [X{iChain},Y{iChain}] = slicesamplebnd(fun,zeros(1,d),nmax,10*ones(1,d),[],[],struct('Adaptive',false,'Display','off','Thin',10));
            Y{iChain} = Y{iChain} - lnz + 100;
        end
    end
    
    if i == 0
        [~,vbmodel] = vbgmmfit(X',30,prior,options);
        vbcornerplot(X,[],[],[min(X);max(X)],[],vbmodel);
    end        

    % Y = Y + randn(size(Y));

    
    [logZ(i,:),temps,output] = vbgmmnormconst(3,X,Y,LB,UB);
    output
    if ~isempty(LB) && ~isempty(UB)
        [logZu(i,:),tempciu] = vbgmmnormconst(3,X,Y,[],[]);
    end
    if isempty(logZu)
        logZ(i,:)
    else
        [logZ(i,:),logZu(i,:)]
    end
    temps
    slogZ(i) = temps;
    
    % logZclo(i,:) = tempci(:,2);
    % logZchi(i,:) = tempci(:,3);
    % logZclou(i,:) = tempciu(:,2);
    % logZchiu(i,:) = tempciu(:,3);
    
    % [mu,vv,hyp] = bayesquad(X,Y,LB,UB);
    % [mu,vv]
    
    % pause
    
    
    % [logZ; logZs]
end
fprintf('\n');
lnf = 100;

mean(sqrt(bsxfun(@plus,logZ,0).^2))
% mean(abs(logZ) < 0.6745*logZs,1)
mean(bsxfun(@ge, lnf, logZclo) & bsxfun(@le, lnf, logZchi),1)
mean(sqrt(bsxfun(@plus,logZu,0).^2))
mean(bsxfun(@ge, lnf, logZclou) & bsxfun(@le, lnf, logZchiu),1)
