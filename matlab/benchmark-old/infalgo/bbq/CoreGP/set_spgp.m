function gp = set_spgp(meanfn_name, gp, X_data, y_data, num_c, num_hypersamples)
% gp = set_spgp(meanfn_name, gp, X_data, y_data, num_c, num_hypersamples)
% covfn_name can be 'sqdexp', 'matern', 'ratquad', 'poly' or 'prodcompact'
% (all of which can be read about in Rasmussen & Williams) and meanfn_name
% can be 'constant', 'planar' or 'quadratic'. The mean function's
% hyperparameters are all set by performing a least-squares fit.

if nargin<6
    num_hypersamples = 1000;
end
num_hypersamples = max(1, ceil(num_hypersamples));

if isempty(gp) 
    gp = struct();
end
if ~isfield(gp,'hyperparams')   
    gp.hyperparams(1) = ...
        struct('name','dummy',...
        'priorMean',nan,...
        'priorSD',nan,...
        'NSamples',nan,...
        'type',nan);
    
else
    default_vals = struct('priorSD', 1, ...
                        'NSamples', nan, ...
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
end

hps_struct = set_hps_struct(gp);

have_X_data = nargin >= 4 && ~isempty(X_data);
have_y_data = nargin >= 5 && ~isempty(y_data);
create_logNoiseSD = ~isfield(hps_struct,'logNoiseSD') ...
                        && ~isfield(gp,'noisefn');
create_log_w0s = ~isfield(hps_struct,'log_w0s') ...
                        || isempty(hps_struct.log_w0s);
create_log_lambda = ~isfield(hps_struct,'log_lambda')...
                        || isempty(hps_struct.log_lambda);
create_meanfn = ~isempty(meanfn_name) && ...
                        (~isfield(gp,'Mu') || isempty(gp.Mu)) && ...
                        (~isfield(gp,'meanPos') || isempty(gp.meanPos));
update_best_hypersample = isfield(gp, 'hypersamples');
create_w_c = ~update_best_hypersample || ...
    ~isfield(gp.hypersamples,'log_tw_c');
create_X_c = ~update_best_hypersample || ...
    ~isfield(gp.hypersamples,'X_c');
                  
if nargin<5
    if have_X_data
        num_data = size(X_data,1);
        num_c = min(num_data, 100);
    else
        num_c = 100;
    end
end


if have_X_data
    [num_data, num_dims] = size(X_data);

    if size(X_data,1) == 1
        input_scales = X_data;
        input_SD = 2;
    else
        input_scales = std(X_data);
        input_SD = 2;
    end
end

w0s = input_scales.^2;

if have_y_data
    output_scale = std(y_data);
    output_SD = 1;
else
    output_scale = exp(10);
    output_SD = 3;
end

if create_logNoiseSD
%     if have_data
    noise_ind = incr_num_hps(gp);
    gp.logNoiseSDPos = noise_ind;

    gp.hyperparams(noise_ind) = orderfields(...
        struct('name','logNoiseSD',...
        'priorMean',log(0.1*output_scale),...
        'priorSD',output_SD,...
        'NSamples',nan,...
        'type','real'),...
        gp.hyperparams);
%     else
%         disp('Need to specify a prior for logNoiseSD, or include data to create one')
%     end
end
if create_log_w0s
    if have_X_data
        w0_inds = nan(1,num_dims);
        
        for dim = 1:num_dims   
            w0_inds(dim) = incr_num_hps(gp);
            
            gp.hyperparams(w0_inds(dim)) = orderfields(...
                struct('name',['log_w0',num2str(dim)],...
                'priorMean',log(w0s(dim)),...
                'priorSD',input_SD,...
                'NSamples',nan,...
                'type','real'),gp.hyperparams);
        end
        gp.w0_inds = w0_inds;
    else
        disp('Need to specify a prior for log_w0, or include data to create one')
    end
else
    gp.w0_inds = hps_struct.log_w0s;  
    w0_inds = gp.w0_inds;
end
if create_log_lambda
%     if have_data
        lambda_ind = incr_num_hps(gp);
        gp.lambda_ind = lambda_ind;
        
        lambda = output_scale * (prod(2*pi*w0s))^(1/4);
        
        gp.hyperparams(lambda_ind) = orderfields(...
            struct('name','log_lambda',...
            'priorMean',log(lambda),...
            'priorSD',output_SD,...
            'NSamples',nan,...
            'type','real'),...
            gp.hyperparams);
%     else
%         disp('Need to specify a prior for log_lambda, or include data to create one')
%     end
end

if create_meanfn
    switch meanfn_name
        case 'constant'
            gp = set_constant_mean(gp, X_data, y_data, []);
        case 'affine'
            gp = set_affine_mean(gp, X_data, y_data ,[]);
        case 'quadratic'
            gp = set_quadratic_mean(gp, X_data, y_data, []);
        otherwise
            % assume constant.
            gp = set_constant_mean(gp, X_data, y_data, []);
    end
end

gp.Mu = get_mu(gp, 'plain');
gp.DMu_inputs = get_mu(gp, 'sp grad inputs');
gp.diag_sqd_noise = get_diag_sqd_noise(gp, 'plain');


if update_best_hypersample
    [logL, best_ind] = max([gp.hypersamples.logL]);
    best_hypersample = gp.hypersamples(best_ind).hyperparameters;
    
    if create_logNoiseSD
        gp.hyperparams(noise_ind).priorMean = best_hypersample(noise_ind);
        gp.hyperparams(noise_ind).priorSD = 0.5;
    end
    if create_log_w0s
        
        for dim = 1:num_dims  
            gp.hyperparams(w0_inds(dim)).priorMean = ...
                best_hypersample(w0_inds(dim));
            gp.hyperparams(w0_inds(dim)).priorSD = 1.5;            
        end
    end
    if create_log_lambda
        gp.hyperparams(lambda_ind).priorMean = ...
            best_hypersample(lambda_ind);
        gp.hyperparams(lambda_ind).priorSD = 1.5;
    end
    if ~create_X_c
        best_X_c = gp.hypersamples(best_ind).X_c;
        num_best_c = size(best_X_c,1);
    end
    if ~create_w_c
        best_log_tw_c = gp.hypersamples(best_ind).log_tw_c;
        num_best_c = size(best_log_tw_c,1);
    end
    
    gp = rmfield(gp, 'hypersamples');
    
elseif have_y_data && have_X_data
    
    gp.X_data = X_data;
    gp.y_data = y_data;

    Mu = get_mu(gp);
    
    y_data_minus_mu = y_data - Mu([gp.hyperparams.priorMean]', X_data);
    
    [noise_sd,input_scales,output_scale] = ...
        hp_heuristics(X_data,y_data_minus_mu);
    
    w0s = (10*input_scales).^2;
    
    lambda = output_scale * (prod(2*pi*w0s))^(1/4);
    
    if create_logNoiseSD
        gp.hyperparams(noise_ind).priorMean = log(noise_sd);
        gp.hyperparams(noise_ind).priorSD = 0.5;
    end
    if create_log_w0s
        
        for dim = 1:num_dims  
            gp.hyperparams(w0_inds(dim)).priorMean = ...
                log(w0s(dim));
            gp.hyperparams(w0_inds(dim)).priorSD = 1.5;            
        end
    end
    if create_log_lambda
        gp.hyperparams(lambda_ind).priorMean = log(lambda);
        gp.hyperparams(lambda_ind).priorSD = 1.5;
    end
end

num_hyperparams = numel(gp.hyperparams);

if ~isfield(gp,'active_hp_inds')
    active=[];
    for hyperparam = 1:num_hyperparams
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

% fill in the prior means for all hyperparameters that are not
% inv_logistic_w_0's
priorMean_mat = repmat([gp.hyperparams.priorMean], num_hypersamples, 1);
priorMean_cell = mat2cell2d(priorMean_mat, ones(num_hypersamples,1), ...
    num_hyperparams);
[gp.hypersamples(1:num_hypersamples).hyperparameters] = priorMean_cell{:};

log_w_0_mat = nan(num_hypersamples, num_dims);
try
    stream = RandStream('mrg32k3a');
    RandStream.setDefaultStream(stream);
catch
    1;
end
phi = lhsdesign(num_hypersamples-1, num_dims);
% this line ensures the prior mean is included
phi = [phi; 0.5 * ones(1, num_dims)];
for dim = 1:num_dims;
    log_w_0_mat(:, dim) = norminv(phi(:,dim), ...
        gp.hyperparams(w0_inds(dim)).priorMean, ...
        gp.hyperparams(w0_inds(dim)).priorSD);
end
for sample = 1:num_hypersamples
	gp.hypersamples(sample).hyperparameters(w0_inds) = ...
        log_w_0_mat(sample,:);
end

if create_X_c
    if have_X_data 
        if num_c >= num_data
            X_c = X_data;
            num_c = num_data;
        else
            X_c = metrickcentres(X_data, num_c);
        end
    else
        X_c = zeros(num_c, num_dims);
    end

    for sample = 1:num_hypersamples
        gp.hypersamples(sample).X_c = X_c;
    end
    
elseif update_best_hypersample
    if num_c <= num_best_c
        X_c = best_X_c(1:num_c, :);
    elseif have_X_data
        more_X_c = metrickcentres(X_data, num_c-num_best_c);

        X_c = nan(num_best_c + size(more_X_c,1), num_dims);
        X_c(1:num_best_c, :) = best_X_c;
        X_c(num_best_c+1:end, :) = more_X_c;
        
    end

    num_c = size(X_c,1);
    
    for sample = 1:num_hypersamples
        gp.hypersamples(sample).X_c = X_c;
    end

end 

if create_w_c
    
    for sample = 1:num_hypersamples
        w_0 = exp(gp.hypersamples(sample).hyperparameters(w0_inds));

        w_c = repmat(w_0, num_c, 1);

        gp.hypersamples(sample).log_tw_c = ...
            log(bsxfun(@minus, w_c, 0.5 * w_0));   
    end
    
elseif update_best_hypersample

    for sample = 1:num_hypersamples
        w_0 = exp(gp.hypersamples(sample).hyperparameters(w0_inds));

        w_c = repmat(w_0, num_c, 1);

        gp.hypersamples(sample).log_tw_c = ...
            log(bsxfun(@minus, w_c, 0.5 * w_0));

        if num_c <= num_best_c
            gp.hypersamples(sample).log_tw_c = best_log_tw_c(1:num_c,:);
        else
            gp.hypersamples(sample).log_tw_c(1:num_best_c,:) = best_log_tw_c;
        end
    end
end


function num = incr_num_hps(gp)
if ~isfield(gp,'hyperparams') || ...
        strcmpi(gp.hyperparams(1).name,'dummy')
    num = 1;
else
    num = numel(gp.hyperparams)+1;
end
