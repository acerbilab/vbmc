function GP = sample_hyperparameters(GP)
% performs initial sampling of hyperparameters for GP. Formerly known as
% `hyperparams.m'


num_hps = numel(GP.hyperparams);

% Some hyperparameters are not `active'; that is, they are assumed to take
% a pre-assigned fixed value. The indices below are only for active
% hyperparameters.
if ~isfield(GP,'active_hp_inds')
    active=[];
    for hyperparam = 1:num_hps
        if GP.hyperparams(hyperparam).priorSD <=0
            GP.hyperparams(hyperparam).type = 'inactive';
        end
        if ~strcmpi(GP.hyperparams(hyperparam).type,'inactive')
            active=[active,hyperparam];
        else
            GP.hyperparams(hyperparam).NSamples=1;
        end
    end
    GP.active_hp_inds=active;
end

% Deal out samples according to priors if it has not already been done
for hyperparam = 1:num_hps
    type = GP.hyperparams(hyperparam).type;
    
    if (~isfield(GP.hyperparams(hyperparam), 'samples') || ...
			isempty(GP.hyperparams(hyperparam).samples));
		mean = GP.hyperparams(hyperparam).priorMean;
		SD = GP.hyperparams(hyperparam).priorSD;
        NSamples = GP.hyperparams(hyperparam).NSamples;
        switch type
            case 'bounded'
                GP.hyperparams(hyperparam).samples = ...				
                    linspacey(mean - 1 * SD, mean + 1 * SD, ...
									NSamples)';
            case 'real'
                GP.hyperparams(hyperparam).samples = ...				
                    norminv(1/(NSamples+1):1/(NSamples+1):NSamples/(NSamples+1),mean,SD)';
            case 'mixture'
                mixtureWeights = GP.hyperparams(hyperparam).mixtureWeights;
                if size(weights,1) == 1
                    mixtureWeights = mixtureWeights';
                    GP.hyperparams(hyperparam).mixtureWeights = mixtureWeights;
                elseif size(weights,1) ~= 1
                    error(['Mixture Weights for hyperparam number',num2str(hyperparam),'have invalid dimension']);
                end
                samples = nan(NSamples,1);
                cdfs = 1/(NSamples+1):1/(NSamples+1):NSamples/(NSamples+1);
                for i=1:NSamples
                    samples(i) = fsolve(@(x) normcdf(x,mean,SD)*mixtureWeights-cdfs(i),0);
                end
                GP.hyperparams(hyperparam).samples = samples;
            case 'inactive'
                GP.hyperparams(hyperparam).samples = mean;
        end
    else
        [samplesize1,samplesize2] = size(GP.hyperparams(hyperparam).samples);
        if samplesize2==1 && samplesize1>=1
            NSamples = samplesize1;
        elseif samplesize2>1 && samplesize1==1
            GP.hyperparams(hyperparam).samples = GP.hyperparams(hyperparam).samples';
            NSamples = samplesize2;
        else
            error(['Samples for hyperparam number',num2str(hyperparam),'have invalid dimension']);
        end
        
        GP.hyperparams(hyperparam).NSamples = NSamples;
    end
    
    
    

end



samples = allcombs({GP.hyperparams(:).samples});
num_samples = size(samples,1);
samples_cell = mat2cell2d(samples,ones(num_samples,1),num_hps);
[GP.hypersamples(1:num_samples).hyperparameters] = samples_cell{:};
