function [qs,covvy] = track_q_fullfn(XData,XStar,covvy,samples)

num_samples = numel(covvy.hypersamples);
Lvec = nan(num_samples,1);
[logLcell{samples}]=covvy.hypersamples(samples).logL;
Lvec(samples)=exp(cat(1,logLcell{samples}));
Lvec(samples)=Lvec(samples)./sum(Lvec(samples));
[Lwm,LwC,covvy]=weighted_gpmeancov(Lvec,XStar,XData,covvy,'store',samples);

q_mean_star=cat(2,covvy.hypersamples(samples).mean_star)'; % from weighted_gpmeancov
q_SD_star=cat(2,covvy.hypersamples(samples).SD_star)'; % from weighted_gpmeancov

% If we have been asked to make predictions about more than one point, we
% arbitrarily choose to minimise our uncertainty around the first.
pt=Lwm(1);
%pt=Lvec'*q_mean_star(:,1)/sum(Lvec);

qs = nan(num_samples,1);
qs(samples)=normpdf(repmat(pt,length(samples),1),q_mean_star(:,1),q_SD_star(:,1));