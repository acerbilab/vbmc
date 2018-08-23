function [bias,H,H_alt] = enttest

plotflag = nargout == 0;

K = 2;

%w = exp(randn(1,K));
w = ones(1,K);
w = w ./ sum(w);
mu = 2*rand(1,K);
sigma = rand(1,K);

% w, mu, sigma


xx = linspace(min(mu)-max(sigma)*5, max(mu)+max(sigma)*5, 1e5);
dx = xx(2)-xx(1);

if plotflag; hold off; end
q = zeros(size(xx));
for k = 1:K
    z = w(k)*normpdf(xx,mu(k),sigma(k));
    q = q + z;
    %if plotflag
    %    plot(xx, z, 'k-', 'LineWidth', 2); hold on;
    %end
    q_k = normpdf(xx,mu(k),sigma(k));
end

if plotflag
    % plot(xx, -p.*log(p), 'k'); hold on;
    % plot(xx, log(p+1e-6), 'k'); hold on;
end

H = qtrapz(-q.*log(q))*dx;

y = normphi(xx,w,mu,sigma);
if plotflag
    %plot(xx, y, 'k--'); hold on;
    %ymax = max(y);
end

if 0
    H_a = 0;
    H_lb = 0;
    H_ub = 0;
    tic
    for k = 1:K

        mu_star = (mu(k).*sigma.^2 + mu.*sigma(k)^2)./(sigma(k)^2 + sigma.^2);
        sigma_star = sqrt(sigma .* sigma(k)./sqrt(sigma(k)^2 + sigma.^2));
        sigmasum = sqrt(sigma.^2+sigma(k)^2);
        w_star = 1./sqrt(2*pi)./sigmasum.*exp(-0.5*(mu(k)-mu).^2./sigmasum.^2);

        subplot(3,4,k);

        large_flag = sigma > 3*sigma(k);
        w_large = normpdf(mu(k),mu,sigma);
        w_large(~large_flag) = 0;        

        ww = exp(-0.5*(mu(k)-mu).^2./sigmasum.^2);
        %ww = exp(-0.5*(mu(k)-mu).^2./sigma.^2);
        ww(large_flag) = 0;


        wf = sum(w.*ww);
        ww = ww/sum(ww);
        sigma
        ww
        wf


        mu_a = sum(ww.*mu);
        sigma_a = sqrt(sum(ww.*(sigma.^2 + (mu - mu_a).^2)));

    %         ww(large_flag) = 0;
    %         wf = sum(w_star.*ww);
    %         ww = ww/sum(ww);
    %         mu_a = sum(ww.*mu_star);
    %         sigma_a = sqrt(sum(ww.*(sigma_star.^2 + (mu_star - mu_a).^2)));


        large_rf = (1./sqrt(2*pi)/sigma_a + sum(w.*w_large)) ./ (1./sqrt(2*pi)/sigma_a);
        wf = wf * large_rf;

        rescaling = interp1(xx,q,mu_a)/(wf./sqrt(2*pi)/sigma_a);
        wf = wf*rescaling;

        q_a = wf*normpdf(xx,mu_a,sigma_a);

        if plotflag
            plot(xx,q,'-','LineWidth',3,'Color',0.8*[1 1 1]); hold on;
            plot(xx,-q_k.*log(q),'k-','LineWidth',2); hold on;
            plot(xx,-q_k.*log(q_a),'b--','LineWidth',1);        
            plot(xx,q_a,':','LineWidth',3,'Color',[0.4 0.4 1]);

            qexp = q_a.*exp(q);

            plot(xx,qexp/interp1(xx,qexp,mu_a)*interp1(xx,q_a,mu_a),':','LineWidth',3,'Color','r');
            drawnow;
            % pause;
        end

        % -0.5*(x^2 - 2*mu_a*x + mu_a^2)/sigma_a^2
        H_a = H_a + w(k)*(-log(wf) + log(sqrt(2*pi)*sigma_a) + 0.5*(mu(k)^2 + sigma(k)^2 - 2*mu_a*mu(k) + mu_a^2)/sigma_a^2);

        gamma_lk = normpdf(mu,mu(k),sqrt(sigma(k)^2 + sigma.^2));
        H_lb = H_lb - w(k)*log(sum(w.*gamma_lk));
        H_ub = H_ub + 0.5*w(k)*(1 + log(2*pi*sigma_a^2)) - w(k)*log(w(k));
    end
    toc

    [H H_a H_lb H_ub]
    bias = max(H_a,H_lb) - H;
end

if 0
    H_p = 0;
    tic
    y_rbf = zeros(size(xx));
    for k = 1:K
        eta = 1.1774;
        Xtrain = [mu(:); mu(:) + eta*sigma(:); mu(:) - eta*sigma(:)]';

        q_k = normpdf(xx,mu(k),sigma(k));
        q = zeros(size(xx));
        for j = 1:K
            q = q + 1/K*normpdf(xx, mu(j), sigma(j));
        end

        mu_star = (mu(k).*sigma.^2 + mu.*sigma(k)^2)./(sigma(k)^2 + sigma.^2);
        sigma_star = sqrt(sigma .* sigma(k)./sqrt(sigma(k)^2 + sigma.^2));

        if plotflag
            subplot(3,4,k);
            yystar = zeros(size(xx));
            for j = 1:K
                yystar = yystar + 1/K*normpdf(xx,mu_star(j),sigma_star(j));
            end        
            %yy = normphi(xx,w,mu,sigma,k)./(w(k))./normpdf(xx,mu(k),sigma(k));
            %plot(xx, yy, 'k--'); hold on;


    %        yymax = max(yy);
            q_star = zeros(size(xx));
            for j = 1:K
                q_star = q_star + 1/K*normpdf(xx, mu_star(j), sigma_star(j));
            end

            large_flag = sigma > 3*sigma(k);
            w_large = normpdf(mu(k),mu,sigma);
            w_large(~large_flag) = 0;        

            sigmasum = sqrt(sigma.^2+sigma(k)^2);
            ww = exp(-0.5*(mu(k)-mu).^2./sigmasum.^2);
            %ww = exp(-0.5*(mu(k)-mu).^2./sigma.^2);
            ww(large_flag) = 0;
            wf = sum(ww/K);
            ww = ww/sum(ww);
            sigma
            ww
            wf

            mu_a = sum(ww.*mu);
            sigma_a = sqrt(sum(ww.*(sigma.^2 + (mu - mu_a).^2)));

            %large_rf = (1./sqrt(2*pi)/sigma_a + sum(1/K.*w_large)) ./ (1./sqrt(2*pi)/sigma_a);
            %wf = wf * large_rf;

            rescaling = interp1(xx,q,mu_a)/(wf./sqrt(2*pi)/sigma_a);
            wf = wf*rescaling;

            q_a = wf*normpdf(xx,mu_a,sigma_a);


            plot(xx,q,'-','LineWidth',3,'Color',0.8*[1 1 1]); hold on;
            plot(xx,-q_k.*log(q),'k-','LineWidth',2); hold on;
            plot(xx,-q_k.*log(q_a),'b--','LineWidth',1);        
            plot(xx,q_a,':','LineWidth',3,'Color',[0.4 0.4 1]);
            drawnow;
            pause;

            %plot(xx, zz/max(zz)*yymax, 'b-', 'LineWidth', 2); hold on;
            % plot(xx,p/max(p)*yymax,'k-'); hold on;

            hh = abs(-normpdf(xx, mu(k), sigma(k)).*log(p));
            hh(~isfinite(hh)) = 0;
            % plot(xx,hh/max(hh)*yymax,'r-','LineWidth',2);
            ff = normpdf(xx, mu(k), 1.2599*sigma(k));
            % plot(xx,ff/max(ff)*yymax,'k:','LineWidth',2);
        end

        Xtrain = [Xtrain, mu_star, mu_star + eta*sigma_star, mu - eta*sigma_star];
        Ytrain = normphi(Xtrain,w,mu,sigma,k)';

        mu_rbf = mu;
        sigma_rbf = repmat(sigma,[1,1]);

        mu_rbf = [mu_rbf, mu_star];
        sigma_rbf = [sigma_rbf, sigma_star];

        w_rbf = rbfn_train(Xtrain,Ytrain,mu_rbf,sigma_rbf);
        y_rbf = y_rbf + rbfn_eval(xx,w_rbf,mu_rbf,sigma_rbf);

        H_p = H_p + sum(w_rbf.*sigma_rbf)*sqrt(2*pi);
    end
    toc
end

%if plotflag
%    plot(xx, zz/max(zz)*ymax, 'k-', 'LineWidth', 2); hold on;
%end

% if plotflag
%     plot(xx, y_rbf, 'r:'); hold on;
% end

%H
%H_alt = -sum(w.*log(w)) + sum(w.*log(sigma.*sqrt(2*pi*exp(1)))) - H_p;

%bias = H_alt - H;

if 0
    H2 = 0;
    tic
    y_rbf = zeros(size(xx));
    for k = 1:K
        eta = 1.0;
        Xtrain = [mu(:); mu(:) + eta*sigma(:); mu(:) - eta*sigma(:); mu(:) + 2*eta*sigma(:); mu(:) - 2*eta*sigma(:); mu(:) + 3*eta*sigma(:); mu(:) - 3*eta*sigma(:)]';

        mu_star = (mu(k).*sigma.^2 + mu.*sigma(k)^2)./(sigma(k)^2 + sigma.^2);
        sigma_star = (sigma .* sigma(k)./sqrt(sigma(k)^2 + sigma.^2));

        Xtrain = [Xtrain, mu_star, mu_star + eta*sigma_star, mu - eta*sigma_star, mu_star + 2*eta*sigma_star, mu - 2*eta*sigma_star, mu - eta*sigma_star, mu_star + 3*eta*sigma_star, mu - 3*eta*sigma_star];
        Ytrain = norment(Xtrain,w,mu,sigma,k)';

        mu_rbf = mu;
        sigma_rbf = repmat(sigma,[1,1]);

        %mu_rbf = [mu_rbf, mu_star];
        %sigma_rbf = [sigma_rbf, sigma_star];

        w_rbf = rbfn_train(Xtrain,Ytrain,mu_rbf,sigma_rbf);
        y_rbf = y_rbf + rbfn_eval(xx,w_rbf,mu_rbf,sigma_rbf);

        H2 = H2 + sum(w_rbf.*sigma_rbf)*sqrt(2*pi);
    end
    toc
    % 
    % if plotflag
    %     plot(xx, y_rbf, 'b-.'); hold on;
    % end

    H - H2
end




if 1
    H2 = 0;
    eta = 1;
    tic
    y_rbf = zeros(size(xx));
    for k = 1:K
        % Xtrain = [mu(:)]';
        Xtrain = [mu(:); mu(:) + eta*sigma(:); mu(:) - eta*sigma(:); mu(:) + 2*eta*sigma(:); mu(:) - 2*eta*sigma(:); mu(:) + 3*eta*sigma(:); mu(:) - 3*eta*sigma(:)]';

        %mu_star = (mu(k).*sigma.^2 + mu.*sigma(k)^2)./(sigma(k)^2 + sigma.^2);
        %sigma_star = (sigma .* sigma(k)./sqrt(sigma(k)^2 + sigma.^2));

        %Xtrain = [Xtrain, mu_star, mu_star + eta*sigma_star, mu - eta*sigma_star, mu_star + 2*eta*sigma_star, mu - 2*eta*sigma_star, mu - eta*sigma_star, mu_star + 3*eta*sigma_star, mu - 3*eta*sigma_star];
        Ytrain = log(interp1(xx,q,Xtrain))' - log(1e-8)

        mu_rbf = mu;
        sigma_rbf = repmat(sigma,[1,1]);

        %mu_rbf = [mu_rbf, mu_star];
        %sigma_rbf = [sigma_rbf, sigma_star];

        w_rbf = rbfn_train(Xtrain,Ytrain,mu_rbf,sigma_rbf)
        y_rbf = y_rbf + rbfn_eval(xx,w_rbf,mu_rbf,sigma_rbf);

        % H2 = H2 + sum(w_rbf.*sigma_rbf)*sqrt(2*pi);
    end
    toc
    % 
    if plotflag
        plot(xx,log(q),'-','LineWidth',3,'Color',0.8*[1 1 1]); hold on;
        plot(xx, y_rbf + log(1e-8),'b--','LineWidth',3); hold on;
    end

    % H - H2
end

if 0
    tol = 1e-8;
    H3 = 0;
    tic
    y_rbf = zeros(size(xx));
    for k = 1:K
        eta = 1.0;
        % Xtrain = [mu(:); mu(:) + eta*sigma(:); mu(:) - eta*sigma(:); mu(:) + 2*eta*sigma(:); mu(:) - 2*eta*sigma(:); mu(:) + 3*eta*sigma(:); mu(:) - 3*eta*sigma(:)]';
        Xtrain = [mu(:); mu(:) + eta*sigma(:); mu(:) - eta*sigma(:)]';
        
        xmat = [-2.57235211094289 -2.16610675289233 -1.95566143558817 -1.80735419679911 -1.6906216295849 -1.59321881802305 -1.50894385503804 -1.43420015968638 -1.36670697180796 -1.30492263775272 -1.24775385535132 -1.19439566356816 -1.14423726510021 -1.09680356209351 -1.05171725299848 -1.0086733576468 -0.967421566101701 -0.927753685357425 -0.889494507530634 -0.852495034274694 -0.816627360848605 -0.781780752765072 -0.747858594763302 -0.714775988103151 -0.68245783666933 -0.650837306444477 -0.619854573565494 -0.589455797849778 -0.559592274227433 -0.530219725824228 -0.501297710767729 -0.472789120992267 -0.444659755988672 -0.416877957995407 -0.389414297852144 -0.362241302844737 -0.335333219514398 -0.308665805694934 -0.282216147062508 -0.255962494294065 -0.229884117579232 -0.203961175751314 -0.17817459772241 -0.152505974246244 -0.126937458305643 -0.101451672641948 -0.0760316231203884 -0.0506606167658766 -0.0253221834133462 0 0.0253221834133464 0.0506606167658766 0.0760316231203884 0.101451672641948 0.126937458305643 0.152505974246244 0.17817459772241 0.203961175751314 0.229884117579232 0.255962494294065 0.282216147062508 0.308665805694934 0.335333219514398 0.362241302844737 0.389414297852145 0.416877957995407 0.444659755988672 0.472789120992268 0.501297710767729 0.530219725824228 0.559592274227433 0.589455797849778 0.619854573565494 0.650837306444477 0.68245783666933 0.71477598810315 0.747858594763302 0.781780752765073 0.816627360848605 0.852495034274694 0.889494507530634 0.927753685357426 0.967421566101701 1.0086733576468 1.05171725299848 1.09680356209351 1.14423726510021 1.19439566356816 1.24775385535132 1.30492263775272 1.36670697180796 1.43420015968638 1.50894385503804 1.59321881802305 1.6906216295849 1.80735419679911 1.95566143558817 2.16610675289233 2.5723521109429];
        Xtrain = [Xtrain, mu(k) + xmat*eta*sigma(k)];
        % Xtrain = mu(:);

        mu_star = (mu(k).*sigma.^2 + mu.*sigma(k)^2)./(sigma(k)^2 + sigma.^2);
        sigma_star = (sigma .* sigma(k)./sqrt(sigma(k)^2 + sigma.^2));

        idx = cleank(Xtrain,mu(k),5*sigma(k));
        Xtrain = Xtrain(idx);
        
        %Xtrain = [Xtrain, mu_star, mu_star + eta*sigma_star, mu - eta*sigma_star];
        Ytrain = logp(Xtrain,w,mu,sigma,tol)';

        mu_rbf = mu;
        sigma_rbf = sigma;
        
        %mu_rbf = [mu,mu(k)+[-1.5,-0.5,0.5,1.5]*sigma(k)];
        %sigma_rbf = [sigma,sigma(k)*ones(1,4)];

        mu_rbf = [mu_rbf, mu_star];
        sigma_rbf = [sigma_rbf, sigma_star]*sqrt(2*pi);
        
        idx = cleank(mu_rbf,mu(k),5*sigma(k));
        mu_rbf = mu_rbf(idx);
        sigma_rbf = sigma_rbf(idx);
        

        w_rbf = rbfn_train(Xtrain,Ytrain,mu_rbf,sigma_rbf)
        y_rbf = rbfn_eval(xx,max(0,w_rbf),mu_rbf,sigma_rbf);
        
        subplot(4,4,k);
        hold off;
        zz = logp(xx,w,mu,sigma,tol);
        plot(xx,y_rbf,'b:','LineWidth',2); hold on;
        plot(xx,zz,'b-');
        zz2 = normpdf(xx,mu(k),sigma(k));
        plot(xx,zz2*max(zz)/max(zz2),'k-');
        xlim([mu(k)-6*sigma(k),mu(k)+6*sigma(k)]);
        ylim([0,max(zz)]);
        
        H3 = H3 - w(k)*sum(w_rbf.*sigma_rbf.*normpdf(mu_rbf,mu(k),sqrt(sigma(k)^2+sigma_rbf.^2)))*sqrt(2*pi);        
    end
    
    H3 = H3 - log(tol);
    [H,H3]
    toc
    % 
    % if plotflag
    %     plot(xx, y_rbf, 'b-.'); hold on;
    % end

end




end

function idx = cleank(mu,mu0,delta0)

idx = mu <= mu0 + delta0 & mu >= mu0 - delta0;

end


function y = logp(xx,w,mu,sigma,tol)

K = numel(w);
y = zeros(size(xx));
for j = 1:K
    y = y + w(j)*normpdf(xx,mu(j),sigma(j));
end

y = log(y+tol) - log(tol);

end



function y = norment(xx,w,mu,sigma,k)

K = numel(w);
phi_k = zeros(size(xx));
for j = 1:K
    phi_k = phi_k + w(j)*normpdf(xx,mu(j),sigma(j));
end

y = -w(k)*normpdf(xx,mu(k),sigma(k)).*log(phi_k+realmin);

end


function y = normphi(xx,w,mu,sigma,k_range)

K = numel(w);
if nargin < 5 || isempty(k_range); k_range = 1:K; end

y = zeros(size(xx));
for k = k_range
    logphi_k = NaN(K,numel(xx));
    for j = 1:K
        logphi_k(j,:) = log(w(j)) - log(w(k)) + normlogpdf(xx,mu(j),sigma(j))-normlogpdf(xx,mu(k),sigma(k));
    end
    lnZ = max(logphi_k,[],1);
    logphi_k = bsxfun(@minus, logphi_k, lnZ);
    phi_k = log(nansum(exp(logphi_k),1)) + lnZ;
    
    y = y + w(k)*normpdf(xx,mu(k),sigma(k)).*phi_k;
end

end

function [F,rho] = rbfn_eval(X,w,Mu,Sigma)
%RBFNEVAL Evaluate radial basis function network.

if nargin < 4 || isempty(Sigma); Sigma = 1; end

[D,N] = size(X);

D2 = sq_dist(Mu, X);
if isscalar(Sigma)
    D2 = D2/Sigma^2;
else
    D2 = bsxfun(@rdivide,D2,Sigma(:).^2);
end

rho = exp(-0.5*D2);

if isempty(w); F = []; else; F = w*rho; end


end

function [w,Phi] = rbfn_train(Xtrain,Ytrain,Mu,Sigma)
    [~,Phi] = rbfn_eval(Xtrain,[],Mu,Sigma);
    w = ((Phi'+ 1e-6*eye(size(Phi'))) \ Ytrain(:))';
end

