function [eta,d_range,time,rmse] = entmcbench(type,sigma)
%ENTMCBENCH Benchmark of Monte Carlo estimators of the entropy (for Gaussian mixtures)

if nargin < 1 || isempty(type); type = 1; end
if nargin < 2 || isempty(sigma); sigma = 1; end
 
eta = []; d_range = []; time = []; rmse = [];

switch type
    case 0
        entk1_plot(sigma);
    case 1
        [eta,d_range,time,rmse] = entk1_benchmark(sigma);
    case 2
        [eta,d_range,time,rmse] = entk2_benchmark();        
    case 3
        [eta_min,d_range,time,rmse] = entk1isa_benchmark(sigma);
end



end

%--------------------------------------------------------------------------
function entk1_plot(sigma)

lb = -10;
ub = 10;
xx = linspace(lb,ub,1e6);
eta = 1.321;

p = normpdf(xx,0,sigma);
y = p.*abs(log(p+realmin));
plot(xx,y,'k--','LineWidth',2); hold on;
plot(xx,p/max(p)*max(y),'k-','LineWidth',2);
p_eta = normpdf(xx,0,sigma*eta);
plot(xx,p_eta/max(p_eta)*max(y),'b:','LineWidth',2);


end

%--------------------------------------------------------------------------
function [eta_min,d_range,time,rmse] = entk1_benchmark(sigma)
%ENTK1_BENCHMARK Benchmark of Monte Carlo methods for entropy estimation

if nargin < 1 || isempty(sigma); sigma = 1; end

% Single multivariate normal

% Best for D = 1 is eta = 1.321
% Then it falls down linearly in D: eta_best(D) ~ 1 + (eta_best(1) - 1)/D

d_range = [1 2 3 5 10 15 20];
d_range = 10;
Niter = 1e4;
Ns = 1000;

% eta_range = linspace(1,1.4,21);

for iD = 1:numel(d_range)
    D = d_range(iD);
    Htrue = 0.5*D*(1 + log(2*pi*sigma^2));
    nf = 1/(2*pi)^(D/2)/sigma^D;    % Common normalization factor
    eta_range = [1,1 + 0.321/D];
    
    bias = zeros(1,numel(eta_range));
    variance = bias;
    rmse = bias;
    time = bias;
    
    for iEta = 1:numel(eta_range)        
        H = zeros(1,Niter);
        
        eta = eta_range(iEta);
        tstart = tic;
        if eta == 1
            for i = 1:Niter
                Xs = randn(D,Ns)*sigma;
                x2sum = sum(Xs.^2,1)/sigma^2;
                ys = nf*exp(-0.5*x2sum);
                H(i) = -sum(log(ys))/Ns;
            end
        else
            for i = 1:Niter
                Xs = randn(D,Ns)*eta*sigma;
                x2sum = sum(Xs.^2,1)/sigma^2;
                ys = nf*exp(-0.5*x2sum);
                w = ys./(nf/eta^D*exp(-0.5*x2sum/eta^2));
                H(i) = -sum(w.*log(ys))/Ns;
            end
        end
        t = toc(tstart);
        bias(iEta) = Htrue - mean(H);
        variance(iEta) = var(H,1);
        rmse(iEta) = sqrt(mean((H - Htrue).^2));
        time(iEta) = t;
    end
    
    plot(eta_range,rmse,'k-'); hold on;
    ylim([0 2]);
    set(gca,'Yscale','log');
    drawnow;
    
    [~,idx] = min(rmse);
    eta_min(iD) = eta_range(idx);
end
   

end


%--------------------------------------------------------------------------
function [eta_min,d_range,time,rmse] = entk1isa_benchmark(sigma)
%ENTK1ISA_BENCHMARK Benchmark of Monte Carlo methods for entropy estimation

if nargin < 1 || isempty(sigma); sigma = 1; end

% Single multivariate normal

% Best for D = 1 is eta = 1.321
% Then it falls down linearly in D: eta_best(D) ~ 1 + (eta_best(1) - 1)/D

d_range = [1 2 3 5 10 15 20];
d_range = 10;
Niter = 1e4;
Ns = 1000;

for iD = 1:numel(d_range)
    D = d_range(iD);
    Htrue = 0.5*D*(1 + log(2*pi*sigma^2));
    nf = 1/(2*pi)^(D/2)/sigma^D;    % Common normalization factor
    
    bias = 0;
    variance = bias;
    rmse = bias;
    time = bias;
    
    eta = 1;
    H = zeros(1,Niter);        
    tstart = tic;
    for i = 1:Niter
        % First draw
        Xs = randn(D,Ns/2)*eta*sigma;
        x2sum = sum(Xs.^2,1)/sigma^2;
        ys = nf*exp(-0.5*x2sum);
        ys = ys.*abs(log(ys));
        w = ys./exp(-0.5*x2sum/eta^2);
        w = w/sum(w);
        
        % Second draw
        mu_star = sum(bsxfun(@times,w,Xs),2);
        dXs = bsxfun(@minus,Xs,mu_star);
        Sigma_star = bsxfun(@times,w,dXs)*dXs'/(1-sum(w.^2));        
        L = chol(Sigma_star);
        
        % Second draw
        Xs = bsxfun(@plus,(randn(Ns/2,D)*L)',mu_star);        
        x2sum = sum(Xs.^2,1)/sigma^2;
        ys = nf*exp(-0.5*x2sum);
        % ys2 = ys.*log(ys);
        w = ys./mvnpdf(Xs',mu_star',Sigma_star)';
        w = w/sum(w);
                
        H(i) = -sum(w.*log(ys));
    end
    t = toc(tstart);
    bias = Htrue - mean(H);
    variance = var(H,1);
    rmse = sqrt(mean((H - Htrue).^2));
    time = t;
    
    %plot(eta_range,rmse,'k-'); hold on;
    %ylim([0 2]);
    %set(gca,'Yscale','log');
    %drawnow;
    
    %[~,idx] = min(rmse);
    eta_min = [];
end
   

end
%--------------------------------------------------------------------------
function [eta_range,d_range,time,rmse] = entk2_benchmark()
%ENTK2_BENCHMARK Benchmark of Monte Carlo methods for entropy estimation

% Mixture of two multivariate normal

d_range = [1 2 3 5 10 15 20];
% d_range = 1;
d_range = 1;
Niter = 1e5;
Ns = 25;

%eta_range = linspace(1.32,1.324,5);

dmu_range = linspace(0,5,11);
sigma_range = [0.01,0.05,0.1,0.2,0.5,1];

bias = zeros(2,numel(dmu_range));
variance = bias;
rmse = bias;
time = bias;

for iSigma = 1:numel(sigma_range)
    sigmafrac = sigma_range(iSigma);
    for iDmu = 1:numel(dmu_range)
        dmu = dmu_range(iDmu)


        for iD = 1:numel(d_range)
            D = d_range(iD);
            eta_range = [1,1 + 0.321/D];
            Htrue = entk2(D,dmu,sigmafrac);
            nf = 1/(2*pi)^(D/2);            % Common normalization factor


            for iEta = 1:numel(eta_range)        
                H = zeros(1,Niter);

                eta = eta_range(iEta);
                tstart = tic;
                if eta == 1
                    for i = 1:Niter
                        Xs = randn(D,Ns);
                        x2sum = sum(Xs.^2,1);
                        xmu2sum = sum((Xs-dmu).^2,1);
                        ys = nf*(0.5*exp(-0.5*x2sum) + 0.5*exp(-0.5*xmu2sum/sigmafrac^2)/sigmafrac^D);        
                        H(i) = H(i)-0.5*sum(log(ys))/Ns;

                        Xs = randn(D,Ns)*sigmafrac + dmu;
                        x2sum = sum(Xs.^2,1);
                        xmu2sum = sum((Xs-dmu).^2,1);
                        ys = nf*(0.5*exp(-0.5*x2sum) + 0.5*exp(-0.5*xmu2sum/sigmafrac^2)/sigmafrac^D);
                        H(i) = H(i)-0.5*sum(log(ys))/Ns;
                    end
                else
                    for i = 1:Niter
                        Xs = randn(D,Ns)*eta;
                        x2sum = sum(Xs.^2,1);
                        xmu2sum = sum((Xs-dmu).^2,1)/sigmafrac^2;
                        ys = nf*(0.5*exp(-0.5*x2sum) + 0.5*exp(-0.5*xmu2sum)/sigmafrac^D);
                        w = exp(-0.5*x2sum*(1-1/eta^2))*eta^D;
                        H(i) = H(i)-0.5*sum(w.*log(ys))/Ns;

                        Xs = randn(D,Ns)*eta*sigmafrac + dmu;
                        x2sum = sum(Xs.^2,1);
                        xmu2sum = sum((Xs-dmu).^2,1)/sigmafrac^2;
                        ys = nf*(0.5*exp(-0.5*x2sum) + 0.5*exp(-0.5*xmu2sum)/sigmafrac^D);
                        w = exp(-0.5*xmu2sum*(1-1/eta^2))*eta^D;
                        H(i) = H(i)-0.5*sum(w.*log(ys))/Ns; 
                    end
                end
                t = toc(tstart);
                bias(iEta,iDmu) = Htrue - mean(H);
                variance(iEta,iDmu) = var(H,1);
                rmse(iEta,iDmu) = sqrt(mean((H - Htrue).^2));
                time(iEta) = t;
            end

        end


    end

    plot(dmu_range,rmse(1,:),'k-'); hold on;
    plot(dmu_range,rmse(2,:),'k--'); hold on;
    ylim([0 0.5]);
    % set(gca,'Yscale','log');
    drawnow;

    %[~,idx] = min(rmse);
    %eta_min(iD) = eta_range(idx);
end

end

%--------------------------------------------------------------------------
function H = entk2(D,dmu,sigmafrac)
%ENTK2 Entropy of a mixture of two Gaussians.

sd_max = max(1,sigmafrac);

if D == 1
    lb = -10*sd_max;
    ub = dmu + 10*sd_max;
    fun1 = @(x) -0.5*normpdf(x,0,1).*log(0.5*normpdf(x,0,1) + 0.5*normpdf(x,dmu,sigmafrac));
    fun2 = @(x) -0.5*normpdf(x,dmu,sigmafrac).*log(0.5*normpdf(x,0,1) + 0.5*normpdf(x,dmu,sigmafrac));
    
    xx = linspace(lb,ub,1e6);
    i1 = qtrapz(fun1(xx))*(xx(2)-xx(1));
    i2 = qtrapz(fun2(xx))*(xx(2)-xx(1));    
    %i1 = integral(fun1,lb,ub);
    %i2 = integral(fun2,lb,ub);
    H = i1 + i2;
    
else
    % Compute volume element
    n = D-1;
    Vn = pi^(n/2)/gamma(1+n/2);
    
end
    
    
    
    




end
