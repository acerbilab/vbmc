function gp = hyperparams(gp)

% % set up the parallelisation of hyperparams if desired
% if ~isfield(gp,'parallel')
%     try
%         matlabpool;
%         gp.parallel = true;
%     catch
%         gp.parallel = false;
%     end
% end
% 
% if gp.parallel
%     isOpen = matlabpool('size') > 0;
%     if ~isOpen
%         matlabpool;
%     end
% end



num_hps = numel(gp.hyperparams);

% I'm going to store these in gp because we only need to calculate them
% once and they'll be needed a fair bit in bmcparams
% if (~isfield(gp,'samplesMean'))
	gp.samplesMean = cat(2, {gp.hyperparams(:).priorMean});
% end
% 
% if (~isfield(gp,'samplesSD'))
	gp.samplesSD = cat(2, {gp.hyperparams(:).priorSD});
% end

% if (~isfield(gp,'names'))
% 	for hyperparam = 1:num_hps
% 		names{hyperparam} = gp.hyperparams(hyperparam).name;
% 	end
% 	gp.names = names;
% end

if ~isfield(gp,'active_hp_inds')
    active=[];
    for hyperparam = 1:num_hps
        if gp.hyperparams(hyperparam).priorSD <=0
            gp.hyperparams(hyperparam).type = 'inactive';
        end
        if ~strcmpi(gp.hyperparams(hyperparam).type,'inactive')
            active=[active,hyperparam];
        else
            gp.hyperparams(hyperparam).NSamples=1;
        end
    end
    gp.active_hp_inds=active;
end
inactive_inds = setdiff(1:num_hps, gp.active_hp_inds);
for active_ind = 1:length(inactive_inds)
    hyperparam = inactive_inds(active_ind);
    gp.hyperparams(hyperparam).NSamples=1;
    gp.hyperparams(hyperparam).type='inactive';
end


if isfield(gp, 'hypersamples')
    warning('hypersamples already exist')
end
% Deal out samples according to priors if it has not already been done
for hyperparam = 1:num_hps
    type = gp.hyperparams(hyperparam).type;
    
    if (~isfield(gp.hyperparams(hyperparam), 'samples') || ...
			isempty(gp.hyperparams(hyperparam).samples));
		mean = gp.hyperparams(hyperparam).priorMean;
		SD = gp.hyperparams(hyperparam).priorSD;
        NSamples = gp.hyperparams(hyperparam).NSamples;
        switch type
            case 'bounded'
                gp.hyperparams(hyperparam).samples = ...				
                    linspacey(mean - 1 * SD, mean + 1 * SD, ...
									NSamples)';
            case {'active','real'}
                gp.hyperparams(hyperparam).samples = ...				
                    norminv(1/(NSamples+1):1/(NSamples+1):NSamples/(NSamples+1),mean,SD)';
            case 'mixture'
                mixtureWeights = gp.hyperparams(hyperparam).mixtureWeights;
                if size(weights,1) == 1
                    mixtureWeights = mixtureWeights';
                    gp.hyperparams(hyperparam).mixtureWeights = mixtureWeights;
                elseif size(weights,1) ~= 1
                    error(['Mixture Weights for hyperparam number',num2str(hyperparam),'have invalid dimension']);
                end
                samples = nan(NSamples,1);
                cdfs = 1/(NSamples+1):1/(NSamples+1):NSamples/(NSamples+1);
                for i=1:NSamples
                    samples(i) = fsolve(@(x) normcdf(x,mean,SD)*mixtureWeights-cdfs(i),0);
                end
                gp.hyperparams(hyperparam).samples = samples;
            case 'inactive'
                gp.hyperparams(hyperparam).samples = mean;
        end
    else
        [samplesize1,samplesize2] = size(gp.hyperparams(hyperparam).samples);
        if samplesize2==1 && samplesize1>=1
            NSamples = samplesize1;
        elseif samplesize2>1 && samplesize1==1
            gp.hyperparams(hyperparam).samples = gp.hyperparams(hyperparam).samples';
            NSamples = samplesize2;
        else
            error(['Samples for hyperparam number',num2str(hyperparam),'have invalid dimension']);
        end
        
        gp.hyperparams(hyperparam).NSamples = NSamples;
    end
end


samples = allcombs({gp.hyperparams(:).samples});
num_samples = size(samples,1);
samples_cell = mat2cell2d(samples,ones(num_samples,1),num_hps);
[gp.hypersamples(1:num_samples).hyperparameters] = samples_cell{:};
gp.hyperparams = rmfield(gp.hyperparams,'samples');


% for i = 1:num_samples
% 	gp.hypersamples(i).hyperparameters = samples(i,:);
% end
