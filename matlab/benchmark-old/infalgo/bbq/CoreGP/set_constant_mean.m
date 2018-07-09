function gp = set_constant_mean(gp, XData, YData, num_samples)
% Set up gp with a quadratic prior mean. It can be called in four ways:
% gp = set_constant_mean(gp)
% sets the mean constant to zero.
% gp = set_constant_mean(gp, num_samples)
% sets a zero mean, 1 SD prior for the constant mean hyperparameter,
% with num_samples taken from each.
% gp = set_constant_mean(gp, XData, YData)
% takes a least-squares fit for the supplied data and sets an appropriate delta function
% prior for the constant mean hyperparameter.
% gp = set_constant_mean(gp, XData, YData, num_samples)
% takes a least-squares fit for the supplied data and uses it to give the
% mean for the prior for the constant mean hyperparameter. The SD
% taken is also estimated from the least-squares fit.

if nargin == 1 
    num_samples = 1;
    type = 'inactive';
    
    const = 0;
    const_priorSD = 10;
elseif nargin == 2 || isempty(YData)
    num_samples = XData;
    type = 'real';
    
    const = 0;
    const_priorSD = 10;
elseif nargin>2
    num_data = length(YData);
    const = mean(YData);

    if nargin>3 && (isempty(num_samples) || isnan(num_samples) )
        
        type = 'inactive';
        const_priorSD = 1;
        
        
    elseif nargin <=3
        num_samples = 1;
        type = 'inactive';
        const_priorSD = 1;
    else
        type = 'real';
        
        error = @(x) sqrt(sum((YData - x*ones(num_data,1)).^2));
        best_error = error(const);
        objective = @(x) best_error*exp(1)-error(x);
        upper_const = fsolve(objective,const);
          
        const_priorSD = upper_const - const;
    end
end

if isfield(gp, 'hyperparams')
    meanPos = numel(gp.hyperparams)+1;

    gp.hyperparams(meanPos) = orderfields(...
    struct('name','MeanConst',...
    'priorMean',const,...
    'priorSD',const_priorSD,...
    'NSamples',num_samples,...
    'type',type),...
    gp.hyperparams);

else
    meanPos = 1;

    gp.hyperparams(meanPos) = ...
    struct('name','MeanConst',...
    'priorMean',const,...
    'priorSD',const_priorSD,...
    'NSamples',num_samples,...
    'type',type);

end
gp.meanPos = meanPos;

