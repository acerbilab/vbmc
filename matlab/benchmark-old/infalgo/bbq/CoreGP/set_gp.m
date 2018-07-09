function gp = set_gp(covfn_name, meanfn_name, gp, X_data, y_data, num_hypersamples)
% gp = set_gp(covfn_name, meanfn_name, gp, X_data, y_data,
% num_hypersamples)
% covfn_name can be 'sqdexp', 'matern', 'ratquad', 'poly' or 'prodcompact'
% (all of which can be read about in Rasmussen & Williams) and meanfn_name
% can be 'constant', 'planar' or 'quadratic'. The mean function's
% hyperparameters are all set by performing a least-squares fit.

% define optional input arguments
% =========================================================================
if nargin<6
    num_hypersamples = 1000;
end
num_hypersamples = max(1, ceil(num_hypersamples));

if isempty(gp) 
    gp = struct();
end

no_hyperparams = ~isfield(gp,'hyperparams');
if no_hyperparams
    num_existing_samples = 1;
    num_existing_hps = 0;
    
    gp.hyperparams(1) = ...
        struct('name','dummy',...
        'priorMean',nan,...
        'priorSD',nan,...
        'NSamples',nan,...
        'type',nan);
    
    grid_num_hypersamples = num_hypersamples;
    
else
    % insert default values should any fields be missing from
    % gp.hyperparams
    default_vals = struct('priorSD', 1, ...
                        'NSamples', 1, ...
                        'type', 'inactive');

    names = fieldnames(default_vals);
    for i = 1:length(names);
        name = names{i};
        for ind = 1:numel(gp.hyperparams)
            if (~isfield(gp.hyperparams(ind), name)) ...
                    || isempty(gp.hyperparams(ind).(name))
                gp.hyperparams(ind).(name) = default_vals.(name);
            end
        end
    end

    num_existing_samples = prod([gp.hyperparams.NSamples]);
    num_existing_hps = numel(gp.hyperparams);
    
    grid_num_hypersamples = max(num_existing_samples, num_hypersamples);
end
hps_struct = set_hps_struct(gp);

if ~isfield(gp, 'active_hp_inds')
    active=[];
    for hyperparam = 1:numel(gp.hyperparams)
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

if isfield(gp,'cov_fn')
    gp.covfn = gp.cov_fn;
    gp = rmfield(gp, 'cov_fn');
end

have_X_data = nargin >= 4 && ~isempty(X_data);
have_y_data = nargin >= 5 && ~isempty(y_data);
create_logNoiseSD = (~isfield(hps_struct,'logNoiseSD')  ...
                        || ismember(hps_struct.logNoiseSD, ...
                                gp.active_hp_inds)) ...
                            && ~isfield(gp,'noisefn');
create_logInputScales = ~isfield(hps_struct,'logInputScales') ...
                        || any(ismember(hps_struct.logInputScales, ...
                                gp.active_hp_inds));
create_logOutputScale = ~isfield(hps_struct,'logOutputScale')...
                        || ismember(hps_struct.logOutputScale, ...
                                gp.active_hp_inds);
create_meanParams = ~isfield(hps_struct,'mean_inds')...
                        || isempty(hps_struct.mean_inds) ...
                        || any(ismember(hps_struct.mean_inds, ...
                                gp.active_hp_inds));
create_covfn = ~isempty(covfn_name) && (~isfield(gp,'covfn_name')...
                || ~all(strcmpi(covfn_name,gp.covfn_name)))...
                        || ~isfield(gp,'covfn')...
                        || isempty(gp.covfn);
create_meanfn = ~isempty(meanfn_name)&& ... % don't already have a mean function
         ( ~isfield(gp,'meanfn_name') || ~all(strcmpi(meanfn_name,gp.meanfn_name)) )...
         ||( ~isfield(gp,'meanfn') || isempty(gp.meanfn) ) ...
             &&( ~isfield(gp,'meanPos') || isempty(gp.meanPos) );
update_best_hypersample = isfield(gp, 'hypersamples');



if have_X_data
    num_dims = size(X_data,2);
    num_hps_to_create = ...
        create_logNoiseSD + ...
        num_dims*create_logInputScales + ...
        create_logOutputScale + ...
        create_meanParams;

    num_samples = factor_in_odds(grid_num_hypersamples/num_existing_samples,num_hps_to_create);
    if size(X_data,1) == 1
        input_scales = X_data;
        input_SD = 2;
    else
        input_scales = std(X_data);
        input_SD = 2;
    end
end

if have_y_data
    output_scale = std(y_data);
    output_SD = 1;
else
    output_scale = exp(10);
    output_SD = 3;
end

if create_logNoiseSD
%     if have_data

    if isfield(hps_struct,'logNoiseSD')
        noise_ind = hps_struct.logNoiseSD;
    else
        noise_ind = incr_num_hps(gp);
        
        gp.hyperparams(noise_ind) = orderfields(...
        struct('name','logNoiseSD',...
        'priorMean',log(0.1*output_scale),...
        'priorSD',output_SD,...
        'NSamples',num_samples(noise_ind-num_existing_hps),...
        'type','real'),...
        gp.hyperparams);
    end

    
    gp.logNoiseSDPos = noise_ind;


%     else
%         disp('Need to specify a prior for logNoiseSD, or include data to create one')
%     end
end
if create_logInputScales    
    if have_X_data
        inputs_ind = nan(1,num_dims);
        for dim = 1:num_dims   
            
            if isfield(hps_struct,'logInputScales')
                inputs_ind(dim) = hps_struct.logInputScales(dim);
            else
                inputs_ind(dim) = incr_num_hps(gp);
                
                gp.hyperparams(inputs_ind(dim)) = orderfields(...
                struct('name',['logInputScale',num2str(dim)],...
                'priorMean',log(input_scales(dim)),...
                'priorSD',input_SD,...
                'NSamples',num_samples(inputs_ind(dim)-num_existing_hps),...
                'type','real'),gp.hyperparams);
            end

            

        end
        
    else
        disp('Need to specify a prior for logInputScales, or include data to create one')
    end
else
     
end
if create_logOutputScale
%     if have_data

    if isfield(hps_struct,'logOutputScale')
        output_ind = hps_struct.logOutputScale;
    else
        output_ind = incr_num_hps(gp);
        
        gp.hyperparams(output_ind) = orderfields(...
            struct('name','logOutputScale',...
            'priorMean',log(output_scale),...
            'priorSD',output_SD,...
            'NSamples',num_samples(output_ind-num_existing_hps),...
            'type','real'),...
            gp.hyperparams);
    end
        

%     else
%         disp('Need to specify a prior for logOutputScale, or include data to create one')
%     end
end

if create_meanfn
    if have_y_data
        switch meanfn_name
            case 'constant'
                gp = set_constant_mean(gp, X_data, y_data);
            case 'affine'
                gp = set_affine_mean(gp, X_data, y_data);
            case 'quadratic'
                gp = set_quadratic_mean(gp, X_data, y_data);
            otherwise
                meanfn_name = 'constant';
                % assume constant.
                gp = set_constant_mean(gp, X_data, y_data);
        end
    else
        meanfn_name = 'constant';
        gp = set_constant_mean(gp, 1);
    end
    
    gp.meanfn_name = meanfn_name;
end

hps_struct = set_hps_struct(gp);
gp.input_scale_inds = [hps_struct.input_inds{:}]; 
gp.output_scale_ind = hps_struct.logOutputScale; 
noise_ind = hps_struct.logNoiseSD; 


if create_covfn
    % set the covariance function
    gp.covfn_name = covfn_name;
    gp.covfn = @(flag) hom_cov_fn(hps_struct,covfn_name,flag);
    if ~isfield(gp, 'sqd_diffs_cov')
        gp.sqd_diffs_cov = true;
    end
% else %nargin(gp.covfn) == 2
%     try gp.covfn = @(flag) gp.covfn(hps_struct,flag);
%     % we have not initialised the cov fn with hps_struct yet   
%     catch;
%     end
end

if ~isfield(gp, 'sqd_diffs_cov')
    gp.sqd_diffs_cov = false;
end


active=[];
for hyperparam = 1:numel(gp.hyperparams)
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


if have_y_data && have_X_data
if update_best_hypersample
    % set the worst half of hypersamples to exploratory points centred
    % around the current best.
    split_pt = ceil(num_hypersamples/2);
    [logLs, sort_inds] = sort([gp.hypersamples.logL],2,'descend');
    best_hypersamples = vertcat(...
        gp.hypersamples(sort_inds(1:split_pt)).hyperparameters);
    best_hypersample = best_hypersamples(1,:);
    
    for i = 1:length(gp.active_hp_inds);
        ind = gp.active_hp_inds(i);
        gp.hyperparams(ind).priorMean = best_hypersample(ind);
    end
    
    % this actually recreates best_hypersample
    gp = rmfield(gp,'hypersamples');
    gp = create_lhs_hypersamples(gp, split_pt);
    
    for ind = 1:(size(best_hypersamples,1)-1)
        gp.hypersamples(split_pt + ind).hyperparameters = ...
            best_hypersamples(ind,:);
    end
    
else %if (create_logNoiseSD || create_logInputScales || create_logOutputScale)
    
    gp = set_gp_data(gp, X_data, y_data);

    Mu = get_mu(gp);
    
    y_data_minus_mu = y_data - Mu([gp.hyperparams.priorMean]', X_data);
    
    [est_noise_sd,est_input_scales,est_output_scale] = ...
        hp_heuristics(X_data,y_data_minus_mu);

    if create_logNoiseSD
        gp.hyperparams(noise_ind).priorMean = log(est_noise_sd);
        gp.hyperparams(noise_ind).priorSD = 0.5;
    end
    if create_logInputScales
        for dim = 1:num_dims  
            gp.hyperparams(inputs_ind(dim)).priorMean = ...
                log(est_input_scales(dim));
            gp.hyperparams(inputs_ind(dim)).priorSD = 1.5;            
        end
    end
    if create_logOutputScale
        gp.hyperparams(output_ind).priorMean = log(est_output_scale);
        gp.hyperparams(output_ind).priorSD = 1.5;
    end
    
    gp = create_lhs_hypersamples(gp, num_hypersamples);
end
end




function num = incr_num_hps(gp)
if ~isfield(gp,'hyperparams') || ...
        strcmpi(gp.hyperparams(1).name,'dummy')
    num = 1;
else
    num = numel(gp.hyperparams)+1;
end

function factors = factor_in_odds(big_number,num_factors)
big_number = floor(big_number);
num_factors = max(1,num_factors);

if big_number<3^num_factors
    number_of_threes=floor(log(big_number)/log(3));
    factors = ones(num_factors,1);
    factors(2:(1+number_of_threes)) = 3;
else
    odd_num = floor(big_number^(1/(num_factors)));
    if odd_num/2 == floor(odd_num/2)
        odd_num = odd_num-1;
    end
    factors = ones(num_factors,1) * odd_num;
    ind = 1; % this is deliberate, we want to add samples to input scales before the noise
    while prod(factors)<big_number
        ind = ind+1;
        if ind > num_factors
            ind = 1;
        end
        factors(ind) = odd_num;
        if ind == 1
            odd_num = odd_num+2;
        end
    end
    factors(ind) = factors(ind)-2;
end
