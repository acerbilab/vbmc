function [x, y, xstars, predictive_means, predictive_variances, fault_post, ...
    mean_fault_sd, mean_norm_sd, full_hyper] ...
    = fault_bucket(x, y, params)

% parameters

if nargin<3
    params = struct();
end

% covariance function, defaults to matern covariance with \nu = \frac{5}{2}
% plus iid Gaussian noise
if isfield(params,'covariance_name')
    covariance_name = params.covariance_name;
else
    covariance_name = 'covMatern5iso';
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
    try
        matlabpool open
    end

    % data for training hyperparametrs
    training_indices = training_window;
    x_train = x(training_indices);
    y_train = y(training_indices);
    
    meany = mean(y_train);
    
    y_train = y_train - meany;
    
    % starting pt gfor optimisation of output scale
    if isfield(params,'trial_output_scale')
        trial_output_scale = params.trial_output_scale;
    else
        trial_output_scale = std(y_train);
    end

    % starting pt gfor optimisation of input scale
    if isfield(params,'trial_input_scale')
        trial_input_scale = params.trial_input_scale;
    else
        trial_input_scale = std(x_train);
    end

    % starting pt gfor optimisation of noise sd
    if isfield(params,'trial_noise')
        trial_noise = params.trial_noise;
    else
        trial_noise = 1e-4 * trial_output_scale;
    end

    covariance = {'covSum', {covariance_name, 'covNoise'}};
    % train hyperparameters
    full_hyper = minimize(log([trial_input_scale; trial_output_scale; trial_noise]), 'gpr', 20, covariance, x_train(:), y_train(:));
else
    meany = y(1);
end

% store the noise variance; we will handle noise explicitly in a modified
% gpr.m file
normal_noise_sd = exp(full_hyper(end));
hyper = full_hyper(1:(end - 1));

% test covariance, noise removed
covariance = {covariance_name};



% test data
test_indices = setdiff(1:length(x), training_window);
x = x(test_indices);
y = y(test_indices);




K = @(x) feval(covariance{:}, hyper, x);
K2 = @(x, xstar) K_wrapper(covariance{:}, hyper, x, xstar);

y = y - meany;

num_preds = length(x)-lookahead;
predictive_means = nan(num_preds, 1);
predictive_variances = nan(num_preds, 1);
fault_post = nan(num_preds, 1);
xstars =  nan(num_preds, 1);

fault_sd_mean = normal_noise_sd*fault_sd_ratio;

% this gp structure is used solely for hyperparameter management
gp.hyperparams(1) = ...
   struct('name','logFaultNoiseSD',...
       'priorMean',log(fault_sd_mean),...
       'priorSD',1,...
       'NSamples',num_fault_sd_samples,...
       'type','real');
   
gp.hyperparams(2) = ...
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


xa = x(1);
ya = y(1);
Ka = K(xa);

xstar = x(1+lookahead);
Kastar = K2(xa,xstar);
Kstarstar = K(xstar);

for ind = 1:num_samples;

    hypersample = gp.hypersamples(ind).hyperparameters;
    fault_sd = exp(hypersample(1));
    norm_sd = exp(hypersample(2));

    V_fa = Ka+fault_sd^2;
    V_na = Ka+norm_sd^2;

    R_fa = chol(V_fa);
    R_na = chol(V_na);

    log_lik_noise_fa = logmvnpdf(ya,0,V_fa);
    log_lik_noise_na = logmvnpdf(ya,0,V_na);

    % mucking around to avoid numerical issues
    if abs(log_lik_noise_fa-log_lik_noise_na) > 100
        if log_lik_noise_fa > log_lik_noise_na
            hypersample_log_likelihood = ...
                log(fault_prior)+log_lik_noise_fa;
        else
            hypersample_log_likelihood = ...
                log(1-fault_prior)+log_lik_noise_na;
        end
    else
        hypersample_log_likelihood = ...
            log(...
            fault_prior * exp(log_lik_noise_fa) + ...
            (1-fault_prior) * exp(log_lik_noise_na)...
            );
    end

    gp.hypersamples(ind).logL = ...
    hypersample_log_likelihood;

    post_noise_fa = ...
    fault_prior * exp(log_lik_noise_fa - hypersample_log_likelihood);
    gp.hypersamples(ind).post_noise_fa = post_noise_fa;

    V_a = (1-post_noise_fa)^(-1)*(post_noise_fa)^(-1)*...
           V_fa *(((post_noise_fa)^(-1)*V_fa + (1-post_noise_fa)^(-1)*V_na)\V_na);

    R_a = chol(V_a);
    gp.hypersamples(ind).R_a = R_a;

    D_a = (R_a)'\ya;
    gp.hypersamples(ind).D_a = D_a;

    T_astar = (R_a)'\Kastar;
    
    

    
    D_fa = (R_fa)'\ya;
    D_na = (R_na)'\ya;
    
    invW_fa = R_fa\D_fa;
    invW_fa = invW_fa*invW_fa';
    invW_na = R_na\D_na;
    invW_na = invW_na*invW_na';
    
    invW_a = post_noise_fa*invW_fa + (1-post_noise_fa)*invW_na;
    gp.hypersamples(ind).invW_a = invW_a;

    means(ind) = T_astar'*D_a;
    vars(ind) = Kstarstar  ...
                - T_astar'*T_astar ...
                + Kastar'*invW_a*Kastar ...
                - means(ind)^2;
end

rho = weights(gp);
rho = rho';

xstars(1) = xstar;
predictive_means(1) = rho*means;
predictive_variances(1) = rho*(vars+means.^2) - predictive_means(1)^2;
fault_post(1) = rho*[gp.hypersamples(:).post_noise_fa]';



% step through remaining data
for i = 2:(length(x) - lookahead)
    % sliding window
    fprintf('.')
    if rem(i,50) == 0
        fprintf('%g\n',i);
    end
 
 
   if length(xa) >= window_length
       % downdate, remembering that we're about to add a point
       to_drop = length(xa)-window_length+1;
       
       xa(to_drop) = [];
       ya(to_drop) = [];
       for ind = 1:num_samples;
           
           R_a = downdatechol(gp.hypersamples(ind).R_a,to_drop);
           gp.hypersamples(ind).R_a = R_a;
           
           D_a = (R_a)'\ya;
            gp.hypersamples(ind).D_a = D_a;
            
            invW_a = R_a\D_a;
            invW_na = invW_a*invW_a';
                gp.hypersamples(ind).invW_a = invW_a;
       end

   end
   xb = x(i);
   yb = y(i);
   
   xab = [xa;xb];
   yab = [ya;yb];
   
   num_ab = length(yab);

   Kb = K(xb);
   Kab = K2(xa,xb);

   xstar = x(i+lookahead);
   Kabstar = K2(xab,xstar);
   Kstarstar = K(xstar);

   for ind = 1:num_samples;
       
        hypersample = gp.hypersamples(ind).hyperparameters;
        fault_sd = exp(hypersample(1));
        norm_sd = exp(hypersample(2));
        
        
        invW_a = gp.hypersamples(ind).invW_a;

        V_fb = Kb+fault_sd^2;
        V_nb = Kb+norm_sd^2;
        
        R_a = gp.hypersamples(ind).R_a;
        D_a = gp.hypersamples(ind).D_a;
        
        T_ab = (R_a)'\Kab;
        T_ab2 = T_ab'*T_ab;
        
        mean_b_a = T_ab'*D_a;
        
        cov_term =  - T_ab2; ...
                    + Kab'*invW_a*Kab...
                    - mean_b_a^2;
        
                % sometimes one of these goes very very slightly negative
        cov_b_a_n = max(V_nb + cov_term,eps);
        cov_b_a_f = max(V_fb + cov_term,eps);

        log_lik_noise_fb = logmvnpdf(yb, mean_b_a, cov_b_a_f);
        log_lik_noise_nb = logmvnpdf(yb, mean_b_a, cov_b_a_n);
        
        % mucking around to avoid numerical issues
        if abs(log_lik_noise_fb-log_lik_noise_nb) > 100
            if log_lik_noise_fb > log_lik_noise_nb
                hypersample_log_likelihood = ...
                    log(fault_prior)+log_lik_noise_fb;
            else
                hypersample_log_likelihood = ...
                    log(1-fault_prior)+log_lik_noise_nb;
            end
        else
            hypersample_log_likelihood = ...
                log(...
                fault_prior * exp(log_lik_noise_fb) + ...
                (1-fault_prior) * exp(log_lik_noise_nb)...
                );
        end

        gp.hypersamples(ind).logL = ...
        hypersample_log_likelihood + gp.hypersamples(ind).logL;


        post_noise_fb = ...
        fault_prior * exp(log_lik_noise_fb - hypersample_log_likelihood);
        gp.hypersamples(ind).post_noise_fa = post_noise_fb;

       
        %M_b = inv(post_noise_fa*inv(mat_f) + (1-post_noise_fa)*inv(mat_n))
        M_b = cov_b_a_f *(((1-post_noise_fb)*cov_b_a_f + (post_noise_fb)*cov_b_a_n)\cov_b_a_n);
       
        Vab = nan(num_ab,num_ab);
        Vab(end,1:end-1) = Kab';
        Vab(1:end-1,end) = Kab;
        Vab(end,end) = M_b + T_ab2;
       
        R_ab = updatechol(Vab,R_a,num_ab);
        gp.hypersamples(ind).R_a = R_ab;
        
        D_ab = updatedatahalf(R_ab,yab,D_a,R_a,num_ab);
        gp.hypersamples(ind).D_a = D_ab;
        
        T_abstar = (R_ab)'\Kabstar;
        
        V_anb = Vab;
        V_anb(end,end) = V_nb;
        R_anb = updatechol(V_anb,R_a,num_ab);
        D_anb = updatedatahalf(R_anb,yab,D_a,R_a,num_ab);
        
        V_afb = Vab;
        V_afb(end,end) = V_fb;
        R_afb = updatechol(V_afb,R_a,num_ab);
        D_afb = updatedatahalf(R_afb,yab,D_a,R_a,num_ab);

        invW_fb = R_afb\D_afb;
        invW_fb = invW_fb*invW_fb';
        invW_nb = R_anb\D_anb;
        invW_nb = invW_nb*invW_nb';

        invW_b = post_noise_fb*invW_fb + (1-post_noise_fb)*invW_nb;
        gp.hypersamples(ind).invW_a = invW_b;

        means(ind) = T_abstar'*D_ab;
        vars(ind) = Kstarstar  ...
                    - T_abstar'*T_abstar ...                    
                    + Kabstar'*invW_b*Kabstar ...
                    - means(ind)^2;
   end

   rho = weights(gp);
   rho = rho';
   
    xstars(i) = xstar;
   predictive_means(i) = rho*means;
   predictive_variances(i) = rho*(vars+means.^2) - predictive_means(i)^2;
   fault_post(i) = rho*[gp.hypersamples(:).post_noise_fa]';

   



   xa = xab;
   ya = yab;
end

used_data = 1:(length(x) - lookahead);
x = x(used_data);
y = y(used_data);

y = y + meany;
predictive_means = predictive_means + meany;

[posterior_hp_means,dummy,mean_sds] = posterior_hp(gp);
mean_fault_sd = mean_sds(1);
mean_norm_sd = mean_sds(2);

function K = K_wrapper(covariance, hyper, xa, xb)

[dummy, K] = feval(covariance, hyper, xa, xb);
