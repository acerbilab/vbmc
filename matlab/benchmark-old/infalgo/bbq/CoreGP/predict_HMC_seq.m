function [YMean,YSD,covvy,closestInd,monitor]=predict_HMC_seq(XStar,XData,YData,num_steps,covvy,params)

if nargin<6 || ~isfield(params,'print')
    params.print=1;%not reassuring dots;
end
no_monitor = nargout<5;

% Initialises hypersamples
covvy=hyperparams(covvy);
NSamples = numel(covvy.hypersamples);
% All hypersamples need to have their gpparams overwritten
covvy.lastHyperSampleMoved=1:NSamples;

lowr.LT=true;
uppr.UT=true;

count=0;

hmc2('state', 0);

for ind=1:num_steps
    
    
    if params.print==1 && ind>0.01*(1+count)*num_steps
        count=count+1;
        ind
    elseif params.print==0 && ( rem(ind,100) == 0)
        fprintf('.');
    end

    likelihood_fn = @(covvy,index) track_likelihood_fullfn(XData,YData,covvy,index);
    covvy = likelihood_fn(covvy,covvy.lastHyperSampleMoved);
    
    % Use ML_ind from previous time-step -improve_bmc_conditioning could
    % potentially moved to after manage_hyper_samples
    
    if ind==1
        % Start our chain off near the peak of the prior
        active_hp_inds = covvy.active_hp_inds;
        last_sample = covvy.hypersamples(ceil(end/2)).hyperparameters(active_hp_inds);
        covvy = manage_hyper_samples_HMC(covvy,likelihood_fn,last_sample);
    else
        covvy = manage_hyper_samples_HMC(covvy,likelihood_fn);
    end
   
    if ~no_monitor
        [log_ML,closestInd] = max([covvy.hypersamples.logL]);
        samples=cat(1,covvy.hypersamples.hyperparameters);
        num_samples=size(samples,1);
        monitor.t(ind).rho=zeros(num_samples,1);
        monitor.t(ind).rho(closestInd)=1;
        monitor.t(ind).hypersamples=cat(1,covvy.hypersamples.hyperparameters);
    end
    
end

[log_ML,closestInd] = max([covvy.hypersamples.logL]);
rho = ones(NSamples,1)/NSamples;
[YMean,wC] = weighted_gpmeancov(rho,XStar,XData,covvy);    
YSD=sqrt((wC)); 