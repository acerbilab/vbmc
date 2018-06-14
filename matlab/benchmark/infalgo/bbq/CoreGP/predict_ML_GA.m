function [YMean,YSD,covvy,closestInd,monitor]=predict_ML(XStar,XData,YData,num_steps,covvy,params)
% num_steps is the number of iterations allowed to the integration machine

if nargin<6 || ~isfield(params,'print')
    params.print=1;%not reassuring dots;
end
no_monitor = nargout<5;

if num_steps>0
    % Initialises hypersamples
    covvy=hyperparams(covvy);
    % All hypersamples need to have their gpparams overwritten
    covvy.lastHyperSampleMoved=1:numel(covvy.hypersamples);

    lowr.LT=true;
    uppr.UT=true;

    count=0;
end

for ind=1:num_steps 
    
    if params.print==1 && ind>0.01*(1+count)*num_steps
        count=count+1;
        ind
    elseif params.print==0 && ( rem(ind,100) == 0)
        fprintf('.');
    end
    
    covvy = track_likelihood_fullfn(XData,YData,covvy,covvy.lastHyperSampleMoved);
    
    [log_ML,closestInd] = max([covvy.hypersamples.logL]);
   
    covvy = manage_hyper_samples_ML(covvy);
   
    if ~no_monitor
        samples=cat(1,covvy.hypersamples.hyperparameters);
        num_samples=size(samples,1);
        monitor.t(ind).rho=zeros(num_samples,1);
        monitor.t(ind).rho(closestInd)=1;
        monitor.t(ind).hypersamples=cat(1,covvy.hypersamples.hyperparameters);
    end
    
end

[log_ML,closestInd] = max([covvy.hypersamples.logL]);
[YMean,wC] = gpmeancov(XStar,XData,covvy,closestInd);
YSD=sqrt(diag(wC)); 