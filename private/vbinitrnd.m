function [vp0_vec,type] = vbinitrnd(Nopts,vp,Knew,Xstar,ystar)
%VBINITRND Generate array of random starting parameters for variational posterior

% XSTAR and YSTAR are usually HPD regions

D = vp.D;
K = vp.K;
lambda0 = vp.lambda;

Nstar = size(Xstar,1);

add_jitter = true;

% Compute moments
%X_mean = mean(X,1);
%X_cov = cov(X);
%[X_R,p] = chol(X_cov);
%if p > 0; X_R = diag(std(X)); end

type = zeros(Nopts,1);

for iOpt = 1:Nopts

    vp0_vec(iOpt) = vp;
    vp0_vec(iOpt).K = Knew;
        
    if iOpt <= Nopts/3  % Start from old variational parameters
        type(iOpt) = 1;
        mu0 = vp.mu;
        sigma0 = vp.sigma;
        if iOpt == 1 % Copy previous parameters verbatim
            add_jitter = false;
        end
        if Knew > vp.K
            % Spawn a new component near an existing one
            for iNew = vp.K+1:Knew
                idx = randi(vp.K);
                mu0(:,iNew) = mu0(:,idx);
                sigma0(iNew) = sigma0(idx);
                mu0(:,iNew) = mu0(:,iNew) + 0.5*sigma0(iNew)*lambda0.*randn(D,1);
                sigma0(iNew) = sigma0(iNew)*exp(0.2*randn());
           end
        end

    elseif iOpt <= Nopts*2/3  % Start from highest-posterior density training points
        type(iOpt) = 2;
        [~,ord] = sort(ystar,'descend');
        mu0 = Xstar(ord(1:Knew),:)';
        sigma0 = sqrt(mean(var(mu0,[],2)./lambda0.^2)/Knew).*exp(0.2*randn(1,Knew));
        if iOpt <= Nopts*2/3+1
            add_jitter = false;                
        end
        if vp.optimize_lambda
            lambda0 = std(Xstar,[],1)';
            lambda0 = lambda0*sqrt(D/sum(lambda0.^2));
        end

    else  % Start from random provided training points
        type(iOpt) = 3;
        ord = randperm(Nstar);
        mu0 = Xstar(ord(1:Knew),:)';
        sigma0 = sqrt(mean(var(mu0,[],2).^2)/Knew)*exp(0.2*randn(1,Knew));
        if vp.optimize_lambda
            lambda0 = std(Xstar,[],1)';
            lambda0 = lambda0*sqrt(D/sum(lambda0.^2));
        end
    end

    if add_jitter
        mu0 = mu0 + bsxfun(@times,sigma0,bsxfun(@times,lambda0,randn(size(mu0))));
        sigma0 = sigma0.*exp(0.2*randn(1,Knew));
        if vp.optimize_lambda
            lambda0 = lambda0.*exp(0.2*randn(D,1));
        end
    end

    vp0_vec(iOpt).mu = mu0;
    vp0_vec(iOpt).sigma = sigma0;
    vp0_vec(iOpt).lambda = lambda0;

end