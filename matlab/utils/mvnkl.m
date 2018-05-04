function [kl1,kl2] = mvnkl(Mu1,Sigma1,Mu2,Sigma2)
%MVNKL Kullback-Leibler divergence between two multivariate normal pdfs.

D = numel(Mu1);

Mu1 = Mu1(:);
Mu2 = Mu2(:);

dmu = Mu2 - Mu1;
detq1 = det(Sigma1);
detq2 = det(Sigma2);
lndet = log(detq2 / detq1);

kl1 = 0.5*(trace(Sigma2\Sigma1) + dmu'*(Sigma2\dmu) - D + lndet);
if nargout > 1
    kl2 = 0.5*(trace(Sigma1\Sigma2) + dmu'*(Sigma1\dmu) - D - lndet);
end