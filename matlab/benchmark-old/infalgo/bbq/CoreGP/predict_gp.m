function [YMean,YSD,gp,closestInd,output]=predict_gp(XStar,...
    gp, params, quad_gp)
% [YMean,YSD,gp,closestInd,output]=predict_gp(XStar,...
%    gp, params, quad_gp)
% num_steps is the number of iterations allowed to the integration machine

if nargin<3 || isempty(params)
    params = struct();
end
if ~isfield(params,'print')
    params.print=false;
end

output=[];

if ~isfield(gp, 'hypersamples')
% Initialises hypersamples
gp = hyperparams(gp);
end

if params.print
    display('Beginning prediction')
    start_time = cputime;
end

if nargin<4
    [dummyVar,closestInd] = max([gp.hypersamples.logL]);
    [YMean,YVar] = posterior_gp(XStar,gp,closestInd,'var_not_cov');
    YSD=sqrt(YVar); 
else
    weights_mat = bq_params(gp, quad_gp);
    hs_weights = weights(gp, weights_mat);
    
    YMean = 0;
    YVar = 0;
    for sample=1:numel(gp.hypersamples)   
        [hs_YMean,hs_YVar] = posterior_gp(XStar,gp,sample,'var_not_cov');
        
        YMean = YMean + hs_weights(sample)*hs_YMean;
        YVar = YVar + hs_weights(sample)*(hs_YVar + hs_YMean.^2);
    end
    YVar = YVar - YMean.^2;
    YSD = sqrt(YVar); 
end


if params.print
    fprintf('Prediction complete in %g seconds\n', cputime-start_time)
end