function sn2 =  estimate_GPnoise(gp)
%ESTIMATE_GPNOISE Estimate GP observation noise at high posterior density.

HPDTop = 0.2;

[N,D] = size(gp.X);

% Subsample high posterior density dataset
[~,ord] = sort(gp.y,'descend');
N_hpd = ceil(HPDTop*N);
X_hpd = gp.X(ord(1:N_hpd),:);
y_hpd = gp.y(ord(1:N_hpd));
if ~isempty(gp.s2)
    s2_hpd = gp.s2(ord(1:N_hpd));
else
    s2_hpd = [];
end

Ncov = gp.Ncov;
Nnoise = gp.Nnoise;
Ns = numel(gp.post);

sn2 = zeros(size(X_hpd,1),Ns);

for s = 1:Ns
    hyp_noise = gp.post(s).hyp(Ncov+(1:Nnoise));
    sn2(:,s) = gplite_noisefun(hyp_noise,X_hpd,gp.noisefun,y_hpd,s2_hpd);
end

sn2 = median(mean(sn2,2));

end