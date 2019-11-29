function [vp0_vec,type_vec] = vbinit_vbmc(type,Nopts,vp,Knew,Xstar,ystar)
%VBINIT Generate array of random starting parameters for variational posterior

% XSTAR and YSTAR are usually HPD regions

D = vp.D;
K = vp.K;

Nstar = size(Xstar,1);

add_jitter = true;

% Compute moments
%X_mean = mean(X,1);
%X_cov = cov(X);
%[X_R,p] = chol(X_cov);
%if p > 0; X_R = diag(std(X)); end

type_vec = type*ones(Nopts,1);
lambda0 = vp.lambda;
mu0 = vp.mu;
w0 = vp.w;

switch type
    case 1      % Start from old variational parameters
        sigma0 = vp.sigma;
    case 2      % Start from highest-posterior density training points
        [~,ord] = sort(ystar,'descend');
        if vp.optimize_mu
            idx_ord = repmat(1:min(Knew,size(Xstar,1)),[1,ceil(Knew/size(Xstar,1))]);
            mu0 = Xstar(ord(idx_ord(1:Knew)),:)';
        end
        if K > 1; V = var(mu0,[],2); else; V = var(Xstar)'; end
        sigma0 = sqrt(mean(V./lambda0.^2)/Knew).*exp(0.2*randn(1,Knew));
    case 3      % Start from random provided training points
        if vp.optimize_mu; mu0 = zeros(D,K); end
        sigma0 = zeros(1,K);
end

for iOpt = 1:Nopts
    vp0_vec(iOpt) = vp;
    vp0_vec(iOpt).K = Knew;
        
    mu = mu0;
    sigma = sigma0;
    lambda = lambda0;
    if vp.optimize_weights; w = w0; end
    
    switch type
        
        case 1      % Start from old variational parameters    
            if iOpt == 1 % Copy previous parameters verbatim
                add_jitter = false;
            end
            if Knew > vp.K
                % Spawn a new component near an existing one
                for iNew = vp.K+1:Knew
                    idx = randi(vp.K);
                    mu(:,iNew) = mu(:,idx);
                    sigma(iNew) = sigma(idx);
                    mu(:,iNew) = mu(:,iNew) + 0.5*sigma(iNew)*lambda.*randn(D,1);
                    if vp.optimize_sigma
                        sigma(iNew) = sigma(iNew)*exp(0.2*randn());
                    end
                    if vp.optimize_weights
                        xi = 0.25 + 0.25*rand();
                        w(iNew) = xi*w(idx);
                        w(idx) = (1-xi)*w(idx);
                    end
                    
               end
            end
            
        case 2      % Start from highest-posterior density training points
            if iOpt == 1
                add_jitter = false;                
            end
            if vp.optimize_lambda
                lambda = std(Xstar,[],1)';
                lambda = lambda*sqrt(D/sum(lambda.^2));
            end
            if vp.optimize_weights
                w = ones(1,Knew)/Knew;
            end

        case 3      % Start from random provided training points
            ord = randperm(Nstar);
            if vp.optimize_mu
                idx_ord = repmat(1:min(Knew,size(Xstar,1)),[1,ceil(Knew/size(Xstar,1))]);                
                mu = Xstar(ord(idx_ord(1:Knew)),:)';
            else
                mu = mu0;
            end
            if K > 1; V = var(mu,[],2); else; V = var(Xstar)'; end

            if vp.optimize_sigma
                sigma = sqrt(mean(V)/Knew)*exp(0.2*randn(1,Knew));
            end
            if vp.optimize_lambda
                lambda = std(Xstar,[],1)';
                lambda = lambda*sqrt(D/sum(lambda.^2));
            end
            if vp.optimize_weights
                w = ones(1,Knew)/Knew;
            end
            
        otherwise
            error('vbinit:UnknownType', ...
                'Unknown TYPE for initialization of variational posteriors.');
    end

    if add_jitter
        if vp.optimize_mu
            mu = mu + bsxfun(@times,sigma,bsxfun(@times,lambda,randn(size(mu))));
        end
        if vp.optimize_sigma
            sigma = sigma.*exp(0.2*randn(1,Knew));
        end
        if vp.optimize_lambda
            lambda = lambda.*exp(0.2*randn(D,1));
        end
        if vp.optimize_weights
            w = w.*exp(0.2*randn(1,Knew));
            w = w/sum(w);
        end
    end

    if vp.optimize_weights
        vp0_vec(iOpt).w = w;
    else
        vp0_vec(iOpt).w = ones(1,Knew)/Knew;        
    end
    if vp.optimize_mu
        vp0_vec(iOpt).mu = mu;
    else
        vp0_vec(iOpt).mu = mu0;        
    end
    vp0_vec(iOpt).sigma = sigma;
    vp0_vec(iOpt).lambda = lambda;

end