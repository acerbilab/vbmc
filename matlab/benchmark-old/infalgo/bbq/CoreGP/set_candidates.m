function covvy = set_candidates(covvy, num_output_candidates)

if ~isfield(covvy,'sc_closeness_num')
    covvy.sc_closeness_num=1.2;%1.2;
end
sc_closeness_num=covvy.sc_closeness_num;

if ~isfield(covvy,'box_nSDs')
    covvy.box_nSDs=3;
end
box_nSDs=covvy.box_nSDs;

if (nargin < 2); num_output_candidates = 6; end


h2s_ind=covvy.ML_hyper2sample_ind;
tilda_h2s_ind=covvy.ML_tilda_hyper2sample_ind;

% the Q stuff does not involve candidates

hyperscales=exp(covvy.hyper2samples(h2s_ind).hyper2parameters);
tilda_hyperscales=exp(covvy.hyper2samples(tilda_h2s_ind).hyper2parameters);

% choose the `biggest' scales as these are more likely to induce
% conditioning problems.
big_hyperscales = max([hyperscales;tilda_hyperscales]);

active_hp_inds=covvy.active_hp_inds;

num_hyperparams = length(hyperscales);
big_scales = big_hyperscales(active_hp_inds);

% Candidates must be well-separated according to both hyperscales and
% tilda_hyperscales;

hypersamples = cat(1,covvy.hypersamples.hyperparameters);
hypersamples = hypersamples(:, active_hp_inds);
already_explored = covvy.explored;
already_explored = already_explored(:, active_hp_inds);
num_candidates = 0;
n = box_nSDs;

while num_candidates < num_output_candidates
    FullMean=[covvy.samplesMean{:}];
    Mean=FullMean(active_hp_inds);
    nSDs=n*[covvy.samplesSD{active_hp_inds}];
    bounds = [Mean - nSDs; Mean + nSDs];

    if size(hypersamples,2)>1 
        num_output_candidates_ff=[]; % have to allow as many as possible because voronoin is pretty lame
    elseif size(hypersamples,2)==1 
        num_output_candidates_ff = num_output_candidates;
    end

    % sc_closeness_num only used if we are in 1D
    [candidates,sc_dists] = find_farthest([hypersamples;already_explored], bounds, num_output_candidates_ff, big_scales,sc_closeness_num); 
    deficit = num_output_candidates-size(candidates,1);
    if deficit>0
        deficit=min(deficit,size(covvy.explored,1));
        covvy.explored(1:deficit,:)=[];
        already_explored = covvy.explored;
        already_explored = already_explored(:, active_hp_inds);
        [candidates,sc_dists] = find_farthest([hypersamples;already_explored], bounds, num_output_candidates, big_scales,sc_closeness_num);
    end

    num_candidates = size(candidates,1);
    n = n+1;
end

