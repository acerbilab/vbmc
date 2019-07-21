function [vpp,lnZ] = vbmc_power(vp,n,cutoff)
%VBMC_POWER Compute power posterior of variational approximation.

if nargin < 3 || isempty(cutoff); cutoff = 1e-5; end
if cutoff < 0; cutoff = 0; end

vpp = vp;
K = vp.K;

if K > 1    
    % For ease of reference in the code
    D = vp.D;
    w = vp.w;
    mu = vp.mu;
    sigma = vp.sigma;
    lambda = vp.lambda;
    
    % Power posterior parameters
    Kp = K^2;
    wp = zeros(1,Kp);
    mup = zeros(D,Kp);
    sigmap = zeros(1,Kp);    
end

switch n
    case 1; lnZ = 0; return;
    case 2
        nf = 1/sqrt(2*pi)^D;

        % First, compute product posterior weights
        idx = 0;        
        for i = 1:K
            for j = 1:K
                idx = idx + 1;
                sigmatilde2 = (sigma(i)^2+sigma(j).^2).*lambda.^2;
                wp(idx) = w(i)*w(j).*nf/prod(sqrt(sigmatilde2))*exp(-0.5*sum((mu(:,i)-mu(:,j)).^2./sigmatilde2,1));
            end
        end
        
        Z = sum(wp);    % Normalization constant
        lnZ = log(Z);        
        wp = wp/Z;
        
        % Throw away components which sum below cutoff
        wp_sorted = sort(wp);
        wp_cum = cumsum(wp_sorted);
        idx_cut = sum(wp_cum < cutoff);
        if idx_cut > 0; w_cutoff = wp_sorted(idx_cut); else; w_cutoff = 0; end
        wp(wp <= w_cutoff) = 0;
        wp = wp/sum(wp);
        
        % Then, compute mean and variance for above-cutoff components only
        idx = 0;
        for i = 1:K
            for j = 1:K
                idx = idx + 1;
                if wp(idx) == 0; continue; end
                mup(:,idx) = (mu(:,i).*sigma(j)^2 + mu(:,j).*sigma(i)^2)./(sigma(i)^2+sigma(j)^2);
                sigmap(idx) = sigma(i)*sigma(j)/sqrt(sigma(i)^2+sigma(j)^2);
            end
        end
    
    otherwise        
        error('vbmc_power:UnsupportedPower',...
            'The power N should be a small positive integer. Currently supported values of N: 1 and 2.');
end

% Keep only nonzero components
keep_idx = wp > 0;
wp_keep = wp(keep_idx);
wp_keep = wp_keep/sum(wp_keep);

vpp.K = sum(keep_idx);
vpp.mu = mup(:,keep_idx);
vpp.sigma = sigmap(keep_idx);
vpp.w = wp_keep;
if isfield(vpp,'temperature') && ~isempty(vpp.temperature)
    vpp.temperature = vpp.temperature/n;
else
    vpp.temperature = 1/n;
end


