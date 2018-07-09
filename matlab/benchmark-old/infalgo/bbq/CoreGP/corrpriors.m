function covvyout=corrpriors(covvy,logLengthScales,corrAngles,Indep)

numhps=numel(covvy.hyperparams);
covvyout=covvy;
for hp=1:numhps
    covvyout.hyperparams(hp).NSDs=2;
end

GD=length(logLengthScales.priorMean);



in_logLengthScales = @(name) isfield(logLengthScales,name) && ~isempty(logLengthScales.(name));

if ~in_logLengthScales('priorMean')
    corrAngles.priorMean=zeros(1,GD);
end
if ~in_logLengthScales('priorSD')
    logLengthScales.priorSD=0.1*abs(logLengthScales.priorMean);
end
if ~in_logLengthScales('NSDs')
    logLengthScales.NSDs=2*ones(1,GD);
end
if ~in_logLengthScales('NSamples')
    logLengthScales.NSamples=3;
end
if ~in_logLengthScales('type')
    if logLengthScales.NSamples == 1
        logLengthScales.type = 'inactive';
    else
        logLengthScales.type = 'real';
    end
end


if Indep
    
    RMean=logLengthScales.priorMean;
    RSD=logLengthScales.priorSD;
    RNSDs=logLengthScales.NSDs;
    
    for r=1:GD
        covvyout.hyperparams(numhps+r)=struct('name',['CorrelationNo',num2str(r)],...
            'priorMean',RMean(r),'priorSD',RSD(r),'NSamples',logLengthScales.NSamples,'type',logLengthScales.type,'NSDs',RNSDs(r));
    end
else

    in_corrangles = @(name) isfield(corrAngles,name) && ~isempty(corrAngles.(name));
    
    rSize=0.5*(GD^2+GD);
    if ~in_corrangles('priorMean')
        corrAngles.priorMean=pi/2*ones(1,rSize-GD);
    end
    if ~in_corrangles('priorSD')
        corrAngles.priorSD=pi*ones(1,rSize-GD);
    end
    if ~in_corrangles('priorSD')
        corrAngles.NSamples=5;
    end
    if ~in_corrangles('NSDs')
        corrAngles.NSDs=corrAngles.priorMean/(pi*2);
    end
    if ~in_corrangles('type')
        if corrAngles.NSamples == 1
            corrAngles.type = 'inactive';
        else
            corrAngles.type = 'real';
        end
    end

    NRSamples=ceil([logLengthScales.NSamples*ones(1,GD),corrAngles.NSamples.^1./(1:(rSize-GD))]);
    % Noting that the sensors' covariance also acts as our proxy for the length
    % scales of the process,expect variation on the order of 1 metre.

    RMean=[logLengthScales.priorMean,corrAngles.priorMean]; % correlations between sensors [angles - so zero correlation = pi/2]
    RSD=[logLengthScales.priorSD,corrAngles.priorSD]; % sds over length scale priors & correlations
    RNSDs=[logLengthScales.NSDs,corrAngles.NSDs]; % Half of the number of SDs over which to sample

    % RIndepSamples=cell(1,rSize);
    % for r=1:rSize
    %     RIndepSamples{r}=normrnd(repmat(RMean(r),NRSamples(r),1),repmat(RSD(r),NRSamples(r),1));
    % end

    % now assign priors as specified above
    for r=1:rSize
        covvyout.hyperparams(numhps+r)=struct('name',['CorrelationNo',num2str(r)],...
            'priorMean',RMean(r),'priorSD',RSD(r),'NSamples',NRSamples(r),'type',corrAngles.type,'NSDs',RNSDs(r));
    end
end

covvyout.num_dims = length(logLengthScales.priorMean);