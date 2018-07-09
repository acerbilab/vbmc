function [dropped,covvy] = improve_bmc_conditioning(covvy)
% A heuristic method of managing potential issues with the conditioning of
% the covariance over the likelihood surface

if ~isfield(covvy,'debug')
    covvy.debug = false;
end
debug = covvy.debug;

if ~isfield(covvy,'ss_closeness_num')
    covvy.ss_closeness_num=1.6;%1.6;
end
ss_closeness_num=covvy.ss_closeness_num;

if ~isfield(covvy,'qq_closeness_num')
    covvy.qq_closeness_num=1;%1
end
qq_closeness_num=covvy.qq_closeness_num;

% any lower than this will give us covariance functions that are
% numerically infinite at their peaks, leading to all kinds of mischief
if ~isfield(covvy,'min_logscale')
    covvy.min_logscale = -6;
end
min_logscale=covvy.min_logscale;

if ~isfield(covvy,'max_logscale')
    covvy.max_logscale = 2.5;
end

%cc_closeness_num=0.5;

% all from previous timestep obviously
h2s_ind=covvy.ML_hyper2sample_ind;
tilda_h2s_ind=covvy.ML_tilda_hyper2sample_ind;
Q_h2s_ind=covvy.ML_Q_hyper2sample_ind;
tildaQ_h2s_ind=covvy.ML_tildaQ_hyper2sample_ind;

if isempty(h2s_ind) || isempty(tilda_h2s_ind) || isempty(Q_h2s_ind) || isempty(tildaQ_h2s_ind)
    % Can't do anything if we don't have any idea about the hyperscales.
    dropped=[];
    covvy.ignoreHyper2Samples=[];
    return
end

hyperscales=exp(covvy.hyper2samples(h2s_ind).hyper2parameters);
tilda_hyperscales=exp(covvy.hyper2samples(tilda_h2s_ind).hyper2parameters);
Q_hyperscales=exp(covvy.hyper2samples(Q_h2s_ind).hyper2parameters);
tildaQ_hyperscales=exp(covvy.hyper2samples(tildaQ_h2s_ind).hyper2parameters);

big_hyperscales = max([[hyperscales;tilda_hyperscales];
    qq_closeness_num/ss_closeness_num*[Q_hyperscales;tildaQ_hyperscales]]);

priorMeans=[covvy.hyperparams.priorMean];
priorSDs=[covvy.hyperparams.priorSD];

active_hp_inds=covvy.active_hp_inds;

samples=cat(1,covvy.hypersamples.hyperparameters);
Nsamples=size(samples,1);
Nhps=size(samples,2);
Nactive_hyperparams = length(active_hp_inds);
Nhyper2samples=numel(covvy.hyper2samples);

%prior_dist=norm([covvy.hyperparams(active_hp_inds).priorSD]./hyperscales(active_hp_inds));



distance_matrix=sqrt(squared_distance(samples, samples, big_hyperscales));

% closer than ss_closeness_num is too close for comfort
[problem_xinds,problem_yinds]=poorly_conditioned(distance_matrix,ss_closeness_num,'upper'); 

dropped=[];
if ~isempty(problem_xinds)
    % A very small expense of computational time here can save our arses if
    % manage_hyper_samples has actually added on a point close to one of our
    % existing candidates.
    covvy=determine_candidates(covvy);
    candidates=cat(1,covvy.candidates.hyperparameters);
    Ncandidates=size(candidates,1);

    sc_distance_matrix=sqrt(squared_distance(samples, candidates, big_hyperscales));
    [sample_inds,bad_candidate_inds]=find(sc_distance_matrix<ss_closeness_num);
    %tilda_sc_distance_matrix=sqrt(squared_distance(samples, candidates, tilda_hyperscales));
    %[sample_inds,bad_candidate_inds]=find(or(sc_distance_matrix<ss_closeness_num, ...
    %                                      tilda_sc_distance_matrix<ss_closeness_num));

    box_nSDs = covvy.box_nSDs;
    lower_bound = priorMeans - box_nSDs*priorSDs;
    upper_bound = priorMeans + box_nSDs*priorSDs;
    outside_box = (candidates < repmat(lower_bound,Ncandidates,1)) +...
                    (candidates > repmat(upper_bound,Ncandidates,1));
    bad_candidate_inds=[bad_candidate_inds;find(any(outside_box,2))];

    far_point_inds = setdiff(1:Ncandidates,bad_candidate_inds);

    j=0;
    while ~isempty(problem_xinds) % We still have problems
        j=j+1;

        % Remove the sample with the lowest logL - if there's a tie,
        % remove the sample that lead to the most problems
        preference_matrix=[histc([problem_xinds;problem_yinds],1:Nsamples),...
                        -[covvy.hypersamples(:).logL]']; % note the negative sign here
        preference_matrix(preference_matrix(:,1)<1,2)=-inf; % don't drop samples that do not give us problems
        [sorted,priority_order]=sortrows(preference_matrix,[2 1]); %sorts ascending

        drop=priority_order(end);
        dropped=[dropped,drop];

        % The indices of the problems that will be solved by dropping this
        % point.
        problem_drop=unique([find(problem_xinds==drop);find(problem_yinds==drop)]);

        problem_xinds(problem_drop)=[];
        problem_yinds(problem_drop)=[];


        while j>length(far_point_inds)
            % Things are pretty hairy if this happens! Maybe just try reducing
            % number of samples.
            if (debug); disp('not enough candidates -  try reducing number of samples.'); end
            covvy=determine_candidates(covvy);
            candidates=cat(1,covvy.candidates.hyperparameters);
            Ncandidates=size(candidates,1);

            sc_distance_matrix=sqrt(squared_distance(samples, candidates, big_hyperscales));
            [sample_inds,bad_candidate_inds]=find(sc_distance_matrix<ss_closeness_num);

            lower_bound = priorMeans - covvy.box_nSDs*priorSDs;
            upper_bound = priorMeans + covvy.box_nSDs*priorSDs;
            outside_box = (candidates < repmat(lower_bound,Ncandidates,1)) +...
                            (candidates > repmat(upper_bound,Ncandidates,1));
            bad_candidate_inds=[bad_candidate_inds;find(any(outside_box,2))];

            far_point_inds = setdiff(1:Ncandidates,bad_candidate_inds);

            j=1;

            if j>length(far_point_inds)
                covvy.box_nSDs = covvy.box_nSDs+1;
            end

            %covvy.hypersamples(drop).hyperparameters=max(samples)+inputscales;
        end

        points = candidates;
        %points=flipud(sortrows(candidates,3));
        far_point=points(far_point_inds(j),:);
        covvy.hypersamples(drop).hyperparameters=far_point;
        covvy.lastHyperSampleMoved = unique([covvy.lastHyperSampleMoved,drop]);

    end
    if (debug); dropped
    end

    samples=cat(1,covvy.hypersamples.hyperparameters);
end

% ssd_cc=separated_squared_distance(candidates,candidates);
% ssd_sc=separated_squared_distance(samples,candidates);
ssd_ss=separated_squared_distance(samples,samples);

scales=[covvy.samplesSD{:}];

problematic=[];
step_size = -covvy.gradient_ascent_step_size;
reduce_step_size=false;

if isfield(covvy,'scale_tilda_likelihood')
    scale_tilda_likelihood = covvy.scale_tilda_likelihood;
else
    scale_tilda_likelihood = true;
end

for h2sample_ind=1:Nhyper2samples
    
    h2sample=covvy.hyper2samples(h2sample_ind).hyper2parameters;
    inputscales=exp(h2sample);
    
    %close_cc=poorly_conditioned(scaled_ssd(ssd_cc,inputscales),cc_closeness_num,'upper');
    close_ss=poorly_conditioned(sqrt(scaled_ssd(ssd_ss,inputscales)),ss_closeness_num,'upper');
    %close_sc=poorly_conditioned(scaled_ssd(ssd_sc,inputscales),ss_closeness_num,'all');
    
    if  ~isempty(close_ss) %|| ~isempty(close_cc) || ~isempty(close_sc)
        
        % of course, what the code below is trying to prevent is exactly
        % what we do want to happen in the event that we have kept both pre
        % and post zoomed samples
%         if ismember(h2sample_ind,[h2s_ind,tilda_h2s_ind,Q_h2s_ind,tildaQ_h2s_ind])...
%                 && ismember(h2sample_ind,setdiff(get_hyper2_samples_to_move(covvy),covvy.ignoreHyper2Samples));
%             % if any of our best hyper2samples end up going problematic, we have
%             % probably just gradient ascended them too far, we reverse the last
%             % gradient ascent step.
% 
%             % shift the to be moved hyper2 samples by one small step of gradient
%             % ascent
%             if scale_tilda_likelihood
%                 gradient=cell2mat(cat(2, covvy.hyper2samples(h2sample_ind).tilda_glogL))';
%             else
%                 gradient=cell2mat(cat(2, covvy.hyper2samples(h2sample_ind).glogL))';
%             end
% 
%             location=h2sample;
%             gradient_full=zeros(1,Nhps);
%             gradient_full(:,active_hp_inds)=gradient;
% 
%             descended = max(gradient_ascent(location, [], gradient_full, step_size, scales),min_logscale);        
%             covvy.hyper2samples(h2sample_ind).hyper2parameters = descended;
%             covvy.lastHyper2SamplesMoved=unique([covvy.lastHyper2SamplesMoved,h2sample_ind]);
%             reduce_step_size = true;
%         else 
            problematic=[problematic,h2sample_ind];
        %end
    end
end

if reduce_step_size
    covvy.gradient_ascent_step_size = 0.95 * covvy.gradient_ascent_step_size;
end

if (debug); problematic
end

%while ~isempty(problematic)
while isempty(setdiff(1:Nhyper2samples,problematic))

    
%     not_problematic=setdiff(1:numel(covvy.hyper2samples),problematic);
%     hyper2samples=cat(1,covvy.hyper2samples(not_problematic).hyper2parameters);
%     
%     try
%     bounds=[min(hyper2samples);max(hyper2samples)];
%     catch
%         1
%     end
%     explore_points = find_farthest(hyper2samples,bounds, length(problematic), scales);
%     
%     num_explore_points=size(explore_points,1);
%     drop_inds=false(num_explore_points,1);
%     for i=1:num_explore_points
%         inputscales=exp(explore_points(i,:));
%     
%         %close_cc=poorly_conditioned(scaled_ssd(ssd_cc,inputscales),cc_closeness_num,'upper');
%         close_ss=poorly_conditioned(scaled_ssd(ssd_ss,inputscales),ss_closeness_num,'upper');
%         %close_sc=poorly_conditioned(scaled_ssd(ssd_sc,inputscales),ss_closeness_num,'all');
% 
%         if ~isempty(close_ss) %|| ~isempty(close_cc) || ~isempty(close_sc)
%             drop_inds(i)=true;
%         end
%     end
%     explore_points(drop_inds,:)=[];
%     
%     num_explore_points=size(explore_points,1);
%     i=0;
    h2sample=problematic(1);
    

    
    inputscales=max(covvy.hyper2samples(h2sample).hyper2parameters-0.01*scales,min_logscale);
    covvy.hyper2samples(h2sample).hyper2parameters=inputscales;
    close_ss=poorly_conditioned(sqrt(scaled_ssd(ssd_ss,exp(inputscales))),ss_closeness_num,'upper');
        %close_sc=poorly_conditioned(scaled_ssd(ssd_sc,inputscales),ss_closeness_num,'all');

    if isempty(close_ss) %|| ~isempty(close_cc) || ~isempty(close_sc)
        covvy.lastHyper2SamplesMoved=unique([covvy.lastHyper2SamplesMoved,problematic(1)]);
        problematic(1)=[];
    end
        
%         i=i+1;
%         if i<=num_explore_points
%             covvy.hyper2samples(h2sample).hyper2parameters=explore_points(i,:);
%         else
%             % if we run out of explore points, just use the ML sample
%             covvy.hyper2samples(h2sample)=covvy.hyper2samples(h2s_ind);
%        end

end

covvy.ignoreHyper2Samples=problematic;

% If I just `blank out' a too-close sample, that is, ignore it, it is
% plausible that eventually we will end with a set that consists entirely
% of such blanked out samples.

function [xinds,yinds]=poorly_conditioned(scaled_distance_matrix,num,flag)
% closer than num scales is too close for comfort
if nargin<3
    flag='all';
end
if strcmp(flag,'upper')
    Nsamples=size(scaled_distance_matrix,1);
    scaled_distance_matrix(tril(true(Nsamples)))=inf;
end
[xinds,yinds]=find(scaled_distance_matrix<num);


function out=separated_squared_distance(A,B)

num_As=size(A,1);
num_Bs=size(B,1);

A_perm=permute(A,[1 3 2]);
A_rep=repmat(A_perm,1,num_Bs);
B_perm=permute(B,[3 1 2]);
B_rep=repmat(B_perm,num_As,1);

out=(A_rep-B_rep).^2;

function out=scaled_ssd(ssd,scales)

[a,b,c]=size(ssd);
tower_scales=repmat(permute(scales.^-2,[1,3,2]),a,b);
out=sum(ssd.*tower_scales,3);
