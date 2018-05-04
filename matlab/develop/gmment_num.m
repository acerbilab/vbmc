function [H,dH] = gmment_num(theta,ell)
%GMMENT_NUM Numerical entropy of Gaussian mixture

D = numel(ell);             % Number of dimensions
K = numel(theta)/(D+1);     % Number of components
MaxSD = 6;                  % Maximum SDs for integration
Nx = ceil(1e3^(1/D));       % Grid size

ell = ell(:);
theta = theta(:);           % THETA is a column vector

% Extract variational parameters from THETA
mu(:,:) = reshape(theta(1:D*K),[D,K]);
sigma(:,1) = exp(theta(D*K+1:end));

H_k = zeros(1,K);

% Loop over mixture components
for k = 1:K
    for d = 1:D
        xvec{d} = linspace(mu(d,k)-ell(d)*sigma(k)*MaxSD,mu(d,k)+ell(d)*sigma(k)*MaxSD,Nx);
        dx(d) = xvec{d}(2)-xvec{d}(1);
    end
    
    Xs = combvec(xvec{:})';
    qs = vbmc_pdf(Xs,theta,ell);
    
    % Single mixture component
    q1s = vbmc_pdf(Xs,theta([(1:D)+(k-1)*D,K*D+k]),ell);
    
    % Integral via Riemannian sum (coarse)
    H_k(k) = sum(q1s .* log(qs)) * prod(dx);        
end

H = -1/K * sum(H_k);

dH = [];    % Gradient computation not supported

end