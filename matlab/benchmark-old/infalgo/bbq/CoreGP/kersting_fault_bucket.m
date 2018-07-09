function [x, y, xstars, predictive_means, predictive_variances, fault_post, ...
    mean_fault_sd, mean_norm_sd, full_hyper] ...
    = brutish_fault_bucket(x, y, params)

% parameters

if nargin<3
    params = struct();
end

% window over which we sample all possible values of faultiness
if isfield(params,'brutish_window_length')
    brutish_window_length = params.brutish_window_length;
else
    brutish_window_length = 7;
end

% covariance function, defaults to matern covariance with \nu = \frac{5}{2}
% plus iid Gaussian noise
if isfield(params,'covariance_name')
    covariance_name = params.covariance_name;
else
    covariance_name = 'matern';
end

% how many time steps to make predictions into the future for online
% prediction
if isfield(params,'lookahead')
    lookahead = params.lookahead;
else
    lookahead = 1;
end


% how many training points to include in a sliding window for online
% prediction
if isfield(params,'window_length')
    window_length = params.window_length;
else
    window_length = 500;
end


% the prior probability that a given observation will be faulty
if isfield(params,'fault_prior')
    fault_prior = params.fault_prior;
else
    fault_prior = 0.01;
end

no_training = isfield(params,'hyperparams');
% hyperparameters for gp
if no_training
    full_hyper = params.hyperparams;
    training_window = [];
else
    % indices of initial training data
    if isfield(params,'training_window')
        training_window = params.training_window;
    else
        training_window = 1:1000;
    end
    
end



% the mean fault noise sd is fault_sd_ratio times the normal noise sd
if isfield(params,'fault_sd_ratio')
    fault_sd_ratio = params.fault_sd_ratio;
else
    fault_sd_ratio = 100;
end


% the number of samples to take in the fault sd
if isfield(params,'num_fault_sd_samples')
    num_fault_sd_samples = params.num_fault_sd_samples;
else
    num_fault_sd_samples = 7;
end

% the number of samples to take in the normal sd
if isfield(params,'num_norm_sd_samples')
    num_norm_sd_samples = params.num_norm_sd_samples;
else
    num_norm_sd_samples = 1;
end



if ~no_training
    % data for training hyperparametrs
    training_indices = training_window;
    x_train = x(training_indices);
    y_train = y(training_indices);
    
    meany = mean(y_train);
    
    y_train = y_train - meany;
    gp = train_gp(covariance_name, 'constant', [], x_train, y_train);

    hps = set_hps_struct(gp);
    best_hypersample = disp_hyperparams(gp);
    
   full_hyper = [best_hypersample(hps.logInputScales); ...
                best_hypersample(hps.logOutputScale); 
                best_hypersample(hps.logNoiseSD)];
else
    meany = mean(y);
end

normal_noise_sd = exp(full_hyper(end));

gp.hyperparams(1) = struct(...
    'name','logInputScale', ...
    'priorMean', full_hyper(1));

gp.hyperparams(2) = struct(...
    'name','logOutputScale', ...
    'priorMean', full_hyper(2));

% summy noisefn, will be overwritten each time before use
gp.noisefn = @(flag) heterosked_noise_fn(0, flag);

gp = set_gp(covariance_name, [], gp);
gp.grad_hyperparams = false;

% test data
test_indices = setdiff(1:length(x), training_window);
x = x(test_indices);
y = y(test_indices);

y = y - meany;


num_preds = length(x)-lookahead;
predictive_means = nan(num_preds, 1);
predictive_variances = nan(num_preds, 1);
fault_post = nan(num_preds, 1);
xstars =  nan(num_preds, 1);

fault_sd_mean = normal_noise_sd*fault_sd_ratio;

% this gp structure is used solely for hyperparameter management

fault_noise_ind = numel(gp.hyperparams)+1;

gp.hyperparams(fault_noise_ind) = ...
   struct('name','logFaultNoiseSD',...
       'priorMean',log(fault_sd_mean),...
       'priorSD',1,...
       'NSamples',num_fault_sd_samples,...
       'type','real');
   
norm_noise_ind = numel(gp.hyperparams)+1;
   
gp.hyperparams(norm_noise_ind) = ...
   struct('name','logNormalNoiseSD',...
       'priorMean',log(normal_noise_sd),...
       'priorSD',1,...
       'NSamples',num_norm_sd_samples,...
       'type','real');
   
gp = hyperparams(gp);
gp = bmcparams(gp);
   
num_samples = numel(gp.hypersamples);



means = nan(num_samples,1);
vars = nan(num_samples,1);
last_a_noise = nan(num_samples,1);

revise_flag = 'overwrite';

% step through remaining data
for i = 2:(length(x) - lookahead)
 % sliding window
    fprintf('.')
    if rem(i,50) == 0
        fprintf('%g\n',i);
    end
 
    xstar = x(i+lookahead);
    
    % a is the segment for which we simply use the ML values of faultiness
    a_inds = 1:(i-brutish_window_length);
    % b is the segment over which we exert brute force
    b_inds = max(1, i+1-brutish_window_length):i;
    
    if length(a_inds)>1
        revise_flag = 'update';
    end
    
    xa = x(a_inds);
    ya = y(a_inds);
    
    xb = x(b_inds);
    yb = y(b_inds);
 
    if length(a_inds) + length(b_inds) >= window_length
        % downdate
        gp = revise_gp([], [], gp, 'downdate', 1);
    end
    

   
    for ind = 1:num_samples;
        
        if ~isempty(a_inds)
            % update R_a;
            gp.noisefn = @(flag) heterosked_noise_fn(last_a_noise(ind), flag);
            gp = revise_gp(xa(end), ya(end), gp, revise_flag, [], ind);
        end
        
       
        hypersample = gp.hypersamples(ind).hyperparameters;
        fault_sd = exp(hypersample(fault_noise_ind));
        norm_sd = exp(hypersample(norm_noise_ind));
        
        % sample over all combinations of faultiness for brute force
        % window
        
        faultiness_combs = ...
            allcombs([zeros(1,length(b_inds));ones(1,length(b_inds))]);
        num_faultiness_combs = size(faultiness_combs,1);
        
        brutish_means = nan(num_faultiness_combs,1);
        brutish_vars = nan(num_faultiness_combs,1);
        brutish_logLs = nan(num_faultiness_combs,1);
        brutish_priors = nan(num_faultiness_combs,1);
        
        parfor faultiness_ind = 1:num_faultiness_combs
            faultiness_vec = faultiness_combs(faultiness_ind,:);
            num_faults = sum(faultiness_vec);
            num_norms = length(faultiness_vec) - num_faults;
            
            noise_vec = ...
                faultiness_vec*fault_sd + (1-faultiness_vec)*norm_sd;
            
            gp_b = gp;
            gp_b.noisefn = @(flag) heterosked_noise_fn(noise_vec, flag);
            gp_b = revise_gp(xb, yb, gp_b, revise_flag, [], ind);
            
            [YMean,YVar] = posterior_gp(xstar, gp_b, ind,'var_not_cov');
            
            brutish_means(faultiness_ind) = YMean;
            brutish_vars(faultiness_ind) = YVar;
            brutish_logLs(faultiness_ind) = gp_b.hypersamples(ind).logL;
            brutish_priors(faultiness_ind) = fault_prior^num_faults *...
                            (1-fault_prior)^num_norms;

        end
        
        % bayes' rule
        scale_factor = max(brutish_logLs);
        scaled_brutish_logLs = brutish_logLs - scale_factor;
        scaled_brutish_posteriors = brutish_priors.*exp(scaled_brutish_logLs);
        
        scaled_hypersample_likelihood = sum(scaled_brutish_posteriors);
        
        gp.hypersamples(ind).logL = log(scaled_hypersample_likelihood) + ...
                                    scale_factor;
        
        brutish_posteriors = scaled_brutish_posteriors./scaled_hypersample_likelihood;
        
        means(ind) = brutish_posteriors'*brutish_means;
        vars(ind) = brutish_posteriors'*(brutish_vars + brutish_means.^2)...
                        - means(ind)^2;
                    
        trail_edge_fault_post = ...
            sum(brutish_posteriors((num_faultiness_combs/2 + 1):...
                                num_faultiness_combs,:));
                  
        if trail_edge_fault_post>0.5
            last_a_noise(ind) = fault_sd;
        else
            last_a_noise(ind) = norm_sd;
        end
    end

                

    rho = weights(gp);
    rho = rho';

    xstars(i) = xstar;
    predictive_means(i) = rho*means;
    predictive_variances(i) = rho*(vars+means.^2) - predictive_means(i)^2;

    % set ML noise variance for point that's just fallen off the back
    % of brutish window
    
end

used_data = 1:(length(x) - lookahead);
x = x(used_data);
y = y(used_data);

y = y + meany;
predictive_means = predictive_means + meany;

[posterior_hp_means,dummy,mean_sds] = posterior_hp(gp);
mean_fault_sd = mean_sds(1);
mean_norm_sd = mean_sds(2);

