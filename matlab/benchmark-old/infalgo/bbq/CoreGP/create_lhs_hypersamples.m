function gp = hyperparams(gp, num_samples)

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

num_active = length(gp.active_hp_inds);

if num_samples>0
priorMean_mat = repmat([gp.hyperparams.priorMean], num_samples, 1);
priorMean_cell = mat2cell2d(priorMean_mat, ones(num_samples,1), ...
    num_hps);
[gp.hypersamples(1:num_samples).hyperparameters] = priorMean_cell{:};

active_hs_mat = nan(num_samples, num_active);
try
    stream = RandStream('mrg32k3a');
    RandStream.setGlobalStream(stream);
catch
    1;
end
phi = lhsdesign(num_samples-1, num_active);
% this line ensures the prior mean is included
phi = [phi; 0.5 * ones(1, num_active)];
for dim = 1:num_active;
    active_hs_mat(:, dim) = norminv(phi(:,dim), ...
        gp.hyperparams(gp.active_hp_inds(dim)).priorMean, ...
        gp.hyperparams(gp.active_hp_inds(dim)).priorSD);
end
for sample = 1:num_samples
	gp.hypersamples(sample).hyperparameters(gp.active_hp_inds) = ...
        active_hs_mat(sample,:);
end
end
