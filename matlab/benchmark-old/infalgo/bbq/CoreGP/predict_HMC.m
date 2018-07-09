function [YMean,YSD,covvy,closestInd]=predict_HMC(XStar,XData,YData,num_steps,covvy,params)

if nargin<6 || ~isfield(params,'print')
    params.print=1;%not reassuring dots;
end

% Initialises hypersamples
covvy = hyperparams(covvy);
num_samples = numel(covvy.hypersamples);
% All hypersamples need to have their gpparams overwritten
covvy.lastHyperSampleMoved=1:num_samples;
likelihood_fn = @(covvy,index) track_likelihood_fullfn(XData,YData,covvy,index);
%covvy = likelihood_fn(covvy,covvy.lastHyperSampleMoved);


hmc2('state', 0);
hmc_options = struct('nsamples',min(num_steps,num_samples),...
        'nomit',max(num_steps-num_samples,0),'display',1,'stepadj',0.04);
hmc_options = hmc2_opt(hmc_options);

% Start our chain off near the peak of the prior
active_hp_inds = covvy.active_hp_inds;
last_sample = covvy.hypersamples(ceil(end/2)).hyperparameters(active_hp_inds);

% Sample
covvy = manage_hyper_samples_HMC(covvy,likelihood_fn,last_sample,hmc_options);
covvy.derivs_cov = false;
covvy = track_likelihood_fullfn(XData,YData,covvy,1:num_samples,'fill_in');


[log_ML,closestInd] = max([covvy.hypersamples.logL]);
rho = ones(num_samples,1)/num_samples;
[YMean,wC] = weighted_gpmeancov(rho,XStar,XData,covvy,'var_not_cov');    
YSD=sqrt((wC)); 