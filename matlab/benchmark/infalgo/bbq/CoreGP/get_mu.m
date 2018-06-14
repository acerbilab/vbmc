function [Mu] = get_mu(gp, flag)

if nargin<2
    flag = 'plain';
end

mean_fun_test = isfield(gp,'meanfn');
if ~mean_fun_test 
    if ~isfield(gp,'meanPos')
        names = {gp.hyperparams.name};
        meanPos = find(cellfun(@(x) strcmpi(x, 'MeanConst'), names));
        if isempty(meanPos)
            meanPos = find(cellfun(@(x) strcmpi(x, 'Mean'), names));
        end
        if isempty(meanPos)
            meanPos = find(cellfun(@(x) strcmpi(x, 'PriorMean'), names));
        end
        gp.meanPos = meanPos;
    else
        meanPos = gp.meanPos;
    end
    % if meanPos is empty, this means we assume a zero mean.
    
    if ~isempty(meanPos)
        Mu = constant_mean_fn(meanPos, flag);
    else
        Mu = @(X,hps) 0; % no matter the flag
    end
else
    Mu = gp.meanfn(flag);
end