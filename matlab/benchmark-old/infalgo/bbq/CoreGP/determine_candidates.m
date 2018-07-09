function covvy = determine_candidates(covvy, num_output_candidates)

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


for h2sample_ind = unique([h2s_ind,tilda_h2s_ind])
    scales = exp(covvy.hyper2samples(h2s_ind).hyper2parameters);
    scales = scales(active_hp_inds);
    
    sc_dists=real(sqrt(squared_distance(hypersamples,candidates,scales)));
    [xinds,yinds]=find(sc_dists<sc_closeness_num);
    
    drop_inds=false(num_candidates,1);
    drop_inds(yinds)=true;
    candidates(drop_inds,:)=[];
    %sc_dists(:,drop_inds)=[];
    num_candidates=size(candidates,1);
    
    sc_dists=real(sqrt(squared_distance(hypersamples,candidates,scales)));
    cc_dists=real(sqrt(squared_distance(candidates,candidates,scales)));

    sumdistances=sum(cc_dists)+sum(sc_dists);
    cc_dists(tril(true(num_candidates)))=inf;

    % the num here can be as low as 0.5 if all we are interested in is using
    % the candidates as candidates. However, don't forget that in
    % improve_bmc_conditioning, we also use candidates as replacements for
    % dropped samples. If we include two or more candidates in this way, we
    % need them to be reasonably well separated, as we're going to get both
    % likelihood and gradient observations at those points.
    
    [problem_xinds,problem_yinds]=find(cc_dists<sc_closeness_num);
    %[problem_xinds,problem_yinds]=poorly_conditioned_pts(cc_dists,sc_closeness_num); 
    

    if ~isempty(problem_xinds)
        
        drop_inds=false(num_candidates,1);
        while ~isempty(problem_xinds) % We still have problems
            % Remove the cand that lead to the most problems - if there's a tie,
            % remove the cand with the lowest total distance from other samples
            preference_matrix=[histc([problem_xinds;problem_yinds],1:num_candidates),...
                                sumdistances'];
            [sorted,priority_order]=sortrows(preference_matrix,[1 2]); %sorts ascending
            priority_order=flipud(priority_order);           
            
            drop=priority_order(1);
            problem_drop=unique([find(problem_xinds==drop);find(problem_yinds==drop)]);
            problem_xinds(problem_drop)=[];
            problem_yinds(problem_drop)=[];
            drop_inds(drop)=true;
        end
        
        % Remove all candidates that are too close to samples
        %[xinds,yinds]=find(sc_dists<sc_closeness_num);
        %[xinds,yinds]=poorly_conditioned_pts(sc_dists,sc_closeness_num); 
        
        candidates(drop_inds,:)=[];
        %sc_dists(:,drop_inds)=[];
        num_candidates=size(candidates,1);
        
    end
    

    
end

candidates=candidates(1:min(num_output_candidates,num_candidates),:);

covvy.candidates = [];
% Just in case we have no candidates
covvy.candidates(1).hyperparameters=zeros(0,num_hyperparams);
covvy.candidates(1).Lsamples=zeros(0,1);

candidates_with_inactive=repmat(FullMean,size(candidates, 1),1);
candidates_with_inactive(:,active_hp_inds)=candidates;

for i = 1:size(candidates, 1)
	covvy.candidates(i).hyperparameters = candidates_with_inactive(i,:);
end

function [xinds,yinds]=poorly_conditioned_pts(scaled_squared_distance_matrix,num)
% closer than two input scales is too close for comfort
[xinds,yinds]=find(scaled_squared_distance_matrix<num);
