function [gp, quad_gp] = train_spgp(gp, X_data, y_data, opt)
% [gp, quad_gp] = train_spgp(gp, X_data, y_data, opt)
% sparse version of train_gp

[num_data, num_dims] = size(X_data);

if nargin<4
    opt = struct();
end

default_opt = struct(...
                    'num_c', min(100, num_data), ...
                    'num_close_c', 20, ...
                    'num_close_d', min(500, num_data), ...
                    'mean_fn', 'constant', ...
                    'derivative_observations', false, ...
                    'num_hypersamples', min(500, 100 * num_dims), ...
                    'optim_time', 60, ...
                    'verbose', false, ...
                    'maxevals_c', 10, ...
                    'maxevals_hs', 10, ...
                    'plots', true, ...
                    'num_passes', 6, ...
                    'force_training', true);
                           
names = fieldnames(default_opt);
for i = 1:length(names);
    name = names{i};
    if (~isfield(opt, name))
      opt.(name) = default_opt.(name);
    end
end
opt.num_close_c = min(opt.num_close_c, opt.num_c);

fprintf('Beginning training of GP, budgeting for %g seconds\n', ...
    opt.optim_time);
start_time = cputime;

if isfield(gp, 'hypersamples')
    hypersamples = gp.hypersamples;
    gp = struct();
    gp.hypersamples = hypersamples;
end


if opt.derivative_observations
    % set_gp assumes a standard homogenous covariance, we don't want to tell
    % it about derivative observations.
    plain_obs = X_data(:,end) == 0;
    
    set_X_data = X_data(plain_obs,1:end-1);
    set_y_data = y_data(plain_obs,:);
    
    
    gp = set_spgp(opt.mean_fn, gp, set_X_data, set_y_data, ...
        opt.num_c, opt.num_hypersamples);
    
    hps_struct = set_hps_struct(gp);
    gp.covfn = @(flag) derivativise(@gp.covfn,flag);
    gp.meanfn = @(flag) wderiv_mean_fn(hps_struct,flag);
    
    gp.X_data = X_data;
    gp.y_data = y_data;

    
else
    gp = set_spgp(opt.mean_fn, gp, X_data, y_data, ...
        opt.num_c, opt.num_hypersamples);
end
% don't want to use likelihood gradients for BMC purposes
gp.grad_hyperparams = false;

full_active_inds = gp.active_hp_inds;
hps_struct = set_hps_struct(gp);

w0_inds = hps_struct.log_w0s;
noise_ind = hps_struct.logNoiseSD;
lambda_ind = hps_struct.log_lambda;

% big_log_noise_sd = gp.hyperparams(lambda_ind).priorMean;
% actual_log_noise_sd = gp.hyperparams(noise_ind).priorMean;


num_hypersamples = numel(gp.hypersamples);
warning('off','revise_spgp:X_c_problems')
tic
[gp] = ...
    revise_spgp(X_data, y_data, gp, 'overwrite', [], ...
    [w0_inds(1), noise_ind]);
hs_eval_time = toc/num_hypersamples;

hypersamples = gp.hypersamples;
gp = rmfield(gp, 'hypersamples');

r_X_data = vertcat(hypersamples.hyperparameters);
r_y_data = vertcat(hypersamples.logL);

[quad_noise_sd, quad_input_scales, quad_output_scale] = ...
    hp_heuristics(r_X_data, r_y_data, 100);
%quad_input_scales = 10 * quad_input_scales;

% only specified in case of early return
quad_gp.quad_noise_sd = quad_noise_sd;
quad_gp.quad_input_scales = quad_input_scales;
quad_gp.quad_output_scale = quad_output_scale;

if opt.optim_time <= 0
    return
end

[max_logL, max_ind] = max(r_y_data);

% initial values for w_c and X_c optimisation below
input_scales = sqrt(exp(...
hypersamples(max_ind).hyperparameters(w0_inds)));

% if these are too large, they'll just be re-estimated within move_c
quad_X_c_scales = 10 * input_scales;
quad_log_tw_c_scales = quad_input_scales;

init_max_logL = max_logL;
fprintf('Initial best log-likelihood: \t%g',init_max_logL);
if opt.verbose
    fprintf(', for ')
    disp_spgp_hps(hypersamples, max_ind,'no_logL');
end
fprintf('\n');

num_c = size(hypersamples(1).X_c, 1);
num_close_c = min(num_c,opt.num_close_c);
num_close_d = min(num_data,opt.num_close_d);

zeroth_X_c_ind = numel(gp.hyperparams);
a_X_c_inds = zeroth_X_c_ind + (1:num_dims*num_close_c);
a_X_c_inds = reshape(a_X_c_inds,num_close_c,num_dims);
a_w_c_inds = zeroth_X_c_ind + num_dims*num_close_c+...
    (1:(num_dims*num_close_c));
a_w_c_inds = reshape(a_w_c_inds,num_close_c,num_dims);

X_close_data = X_data(1:num_close_d,:);
y_close_data = y_data(1:num_close_d,:);
X_close_c = hypersamples(1).X_c(1:num_close_c,:);
log_tw_close_c = ...
    hypersamples(1).log_tw_c(1:num_close_c,:);

% determine how long computing the likelihood of a reduced, local dataset
% takes

a_gp = gp;
a_gp.hypersamples(1) = hypersamples(max_ind);
a_gp.hypersamples(1).X_c = X_close_c;
a_gp.hypersamples(1).log_tw_c = log_tw_close_c;
tic;
revise_spgp(X_close_data, y_close_data, a_gp, 'overwrite', ...
            [], [a_w_c_inds(:,1)', w0_inds(1), noise_ind]);
c_eval_time = toc;
clear a_gp;

maxevals_c = opt.maxevals_c;
maxevals_hs = opt.maxevals_hs;
num_passes = opt.num_passes;

ideal_time = num_hypersamples * (...
                maxevals_c * num_passes * num_c * num_dims * c_eval_time ...
                + maxevals_hs * (num_passes * (num_dims + 1)) ...
                * hs_eval_time...
                );
% the 2 *  here is due to expected speed-up due to
% parallelisation
scale_factor = ... %2 * 
    opt.optim_time / ideal_time;

% set the allowed number of likelihood evaluations
opt.maxevals_hs = ceil(maxevals_hs * scale_factor);
opt.maxevals_c = ceil(maxevals_c * scale_factor);

if opt.maxevals_hs == 1
    warning('insufficient time allowed to train sparse GP, consider decreasing opt.num_hypersamples or increasing opt.optim_time');
    if opt.force_training
        warning('proceeding with minimum possible likelihood evaluations');
        opt.maxevals_hs = 2;
    else
        gp.hypersamples = hypersamples;
        return
    end
end
if opt.maxevals_c == 1
    warning('insufficient time allowed to train X_c and w_c');
    if opt.force_training
        warning('proceeding with minimum possible likelihood evaluations');
        opt.maxevals_c = 2;
    end
end



% if opt.verbose
%     fprintf('%g evals to train w_0, lambda and sigma, %g to train X_c and w_c\n', opt.maxevals_hs, opt.maxevals_c);
% end   

for num_pass = 1:num_passes
    
    if opt.verbose
        fprintf('Pass %g\n', num_pass)
    end
    
    w0_cell = cell(num_hypersamples, 1);
    w0_logL_cell = cell(num_hypersamples, 1);
    
    lambda_cell = cell(num_hypersamples,1);
    lambda_logL_cell = cell(num_hypersamples,1);
    
    parfor hypersample_ind = 1:num_hypersamples
        
        warning('off','revise_spgp:X_c_problems');
        warning('off','revise_spgp:small_num_data');
        
        if opt.verbose
            fprintf('Hyperparameter sample %g\n',hypersample_ind)
        end
        
        big_log_noise_sd = ...
            hypersamples(hypersample_ind).hyperparameters(noise_ind);
        %hypersample.hyperparameters(lambda_ind);
        actual_log_noise_sd = ...
            hypersamples(hypersample_ind).hyperparameters(noise_ind);

        % optimise w_0

        hypersamples(hypersample_ind).hyperparameters(noise_ind) = ...
            big_log_noise_sd;

        w0_cell{hypersample_ind} = cell(1, num_dims);
        w0_logL_cell{hypersample_ind} = cell(1, num_dims);
        for d = 1:num_dims 
            active_hp_inds = [w0_inds(d), noise_ind];
            

            [inputscale_hypersample, w0_mat, w0_logL_mat] = ...
                move_hypersample(...
                hypersamples(hypersample_ind), gp, quad_input_scales, ...
                active_hp_inds, ...
                X_data, y_data, opt);
            
            log_w_0d = ...
                inputscale_hypersample.hyperparameters(w0_inds(d));

            hypersamples(hypersample_ind).hyperparameters(w0_inds(d)) = ...
                log_w_0d;
            
            w0_cell{hypersample_ind}{d} = w0_mat;
            w0_logL_cell{hypersample_ind}{d} = w0_logL_mat;
            
            
            if opt.verbose
                fprintf(', \t for w_0(%g) = %g\n', ...
                    d, exp(log_w_0d));
            end
        end

        hypersamples(hypersample_ind).hyperparameters(noise_ind) = ...
            actual_log_noise_sd;

         % optimise lambda & noise

        active_hp_inds = [lambda_ind, noise_ind];

        [outputscale_hypersample, lambda_mat, lambda_logL_mat] = ...
            move_hypersample(...
                hypersamples(hypersample_ind), gp, quad_input_scales, ...
                active_hp_inds, ...
                X_data, y_data, opt);
            
        lambda_cell{hypersample_ind} = lambda_mat;
        lambda_logL_cell{hypersample_ind} = lambda_logL_mat;
            
%         hypersamples(hypersample_ind).hyperparameters(active_hp_inds) = ...
%                 outputscale_hypersample.hyperparameters(active_hp_inds);
            
%         if opt.verbose
%             fprintf(', \t for lambda = %g\n', ...
%                 exp(log_lambda));
%         end
% 
%         % now do a quick joint optimisation to finish off
% 
%         [hypersample] = ...
%             move_hypersample(...
%                     hypersample, gp, quad_input_scales, ...
%                     full_active_inds, ...
%                     X_data, y_data, opt);

        hypersamples(hypersample_ind) = outputscale_hypersample;
                
        if opt.verbose
            fprintf(', \t for ');
            disp_spgp_hps(hypersamples(hypersample_ind), [], 'no_logL');
            fprintf('\n');
        end

    end
    
    % now we estimate the scale of variation of the likelihood wrt
    % log w_0.
    
    w0_compcell = cat(1,w0_cell{:});
    w0_logL_compcell = cat(1,w0_logL_cell{:});
    quad_noise_sds = nan(num_dims+1,1);
    quad_output_scales = nan(num_dims+1,1);
        
    for d = 1:num_dims
        a_hps_mat = cat(1, w0_compcell{:,d});
        logL_mat = cat(1,w0_logL_compcell{:,d});
        
        sorted_logL_mat = sort(logL_mat);
              
        top_inds = logL_mat >= sorted_logL_mat(round(0.9*end));
        a_hps_mat = a_hps_mat(top_inds,:);
        logL_mat = logL_mat(top_inds,:);

        [quad_noise_sds(d), a_quad_input_scales, quad_output_scales(d)] = ...
            hp_heuristics(a_hps_mat,logL_mat,10);

        quad_input_scales(w0_inds(d)) = a_quad_input_scales(1);
    end
    
    % now we estimate the scale of variation of the likelihood wrt
    % log lambda and log noise sd.
    a_hps_mat = cat(1,lambda_cell{:});
    logL_mat = cat(1,lambda_logL_cell{:});
    
    a_hps_mat = max(a_hps_mat, -100);
    
    sorted_logL_mat = sort(logL_mat);

    top_inds = logL_mat >= sorted_logL_mat(round(0.9*end));
    a_hps_mat = a_hps_mat(top_inds,:);
    logL_mat = logL_mat(top_inds,:);

    [quad_noise_sds(end), a_quad_input_scales, quad_output_scales(end)] = ...
        hp_heuristics(a_hps_mat,logL_mat,10);

    quad_input_scales(lambda_ind) = a_quad_input_scales(1);
    quad_input_scales(noise_ind) = a_quad_input_scales(2);
    
    quad_noise_sd = min(quad_noise_sds);
    quad_output_scale = max(quad_output_scales);
    
    if cputime-start_time > opt.optim_time || num_pass == num_passes
        % need to end here so that we have gp that has been trained on the
        % whole dataset, rather than on a `close' subset
        break
    end

    if num_pass == 1
        quad_X_c_scales = 10 * input_scales;
        quad_log_tw_c_scales = quad_input_scales(w0_inds);
    end
    
    a_hps_cell = cell(num_hypersamples, num_c);
    logL_cell = cell(num_hypersamples, num_c);
    
    for hypersample_ind = 1:num_hypersamples

        big_log_noise_sd = ...
            hypersamples(hypersample_ind).hyperparameters(noise_ind);
        %hypersample.hyperparameters(lambda_ind);
        actual_log_noise_sd = ...
            hypersamples(hypersample_ind).hyperparameters(noise_ind);
        
        if opt.verbose
            fprintf('Hyperparameter sample %g\n',hypersample_ind)
        end
        
        if opt.maxevals_c > 1

        % set X_c and w_c

        % we'll use N_c and N_cd below to select the data points and centers
        % closest to a given center.
        R_c = hypersamples(hypersample_ind).R_c;
        G_cd = hypersamples(hypersample_ind).G_cd;
        N_c = R_c' * R_c;
        N_cd = R_c' * G_cd;
        clear R_c G_cd


        X_c = nan(num_c, num_dims);
        log_tw_c = nan(num_c, num_dims);
        log_w_0_mat = nan(num_c, num_dims); 
        log_noise_mat = nan(num_c, num_dims); 


        parfor a = 1:num_c
            warning('off','revise_spgp:X_c_problems')
            
            corrs_cd = N_cd(a,:);

            [sorted, sort_order] = sort(corrs_cd,2,'descend');
            close_data_inds = sort_order(...
                intlogspace(1,num_data, num_close_d));
            X_close_data = X_data(close_data_inds,:);
            y_close_data = y_data(close_data_inds,:);

            % need to correct for the different w_c's for each c
            corrs_c = N_c(:,a)./sqrt(diag(N_c));

            [sorted, sort_order] = sort(corrs_c,1,'descend');
            close_c_inds = sort_order(...
                intlogspace(1,num_c, num_close_c));
            X_close_c = hypersamples(hypersample_ind).X_c(close_c_inds,:);
            log_tw_close_c = ...
                hypersamples(hypersample_ind).log_tw_c(close_c_inds,:);

           % NB: clearly, the first element of sort order should always be i;
           % that's what we're going to move

            a_gp = gp;
            a_gp.hypersamples = hypersamples(hypersample_ind);
            a_gp.hypersamples.hyperparameters(noise_ind) = ...
                big_log_noise_sd;

            a_gp.hypersamples.X_c = X_close_c;
            a_gp.hypersamples.log_tw_c = ...
                log_tw_close_c;

            X_c_vec = nan(1, num_dims);
            log_tw_c_vec = nan(1, num_dims);
            log_w_0_vec = nan(1, num_dims);
            log_noise_vec = nan(1, num_dims);

            a_hps_cell{hypersample_ind, a} = cell(1,num_dims);
            logL_cell{hypersample_ind, a} = cell(1,num_dims);

            %%%par
            for d = 1:num_dims
                % optimise w_c
                
                [a_X_c_d, a_log_tw_c_d, log_w_0_d, log_noise...
                    a_hps_mat, logL_mat] = ...
                    move_c(d, ...
                    a_gp, quad_X_c_scales, ...
                    quad_log_tw_c_scales, quad_input_scales, ...
                    X_close_data, y_close_data, ...
                    a_X_c_inds, a_w_c_inds, w0_inds, noise_ind, opt);
                X_c_vec(d) = a_X_c_d;
                log_tw_c_vec(d) = a_log_tw_c_d;
                log_w_0_vec(d) = log_w_0_d;
                log_noise_vec(d) = log_noise;
                
                a_hps_cell{hypersample_ind, a}{d} = a_hps_mat(:,[1 2]);
                logL_cell{hypersample_ind, a}{d} = logL_mat;

%                if opt.verbose
%                     fprintf(',\t for X_c(%g,%g) = %f\n', ...
%                         a, d, a_X_c_d) 
%                     fprintf(',\t for w_c(%g, %g) = %f,\t X_c(%g,%g) = %f\n', ...
%                         a, d, a_w_c_d, a, d, a_X_c_d) 
%                end
            end
            
            X_c(a,:) = X_c_vec;
            log_tw_c(a,:) = log_tw_c_vec;
            log_w_0_mat(a,:) = log_w_0_vec;  
            log_noise_mat(a,:) = log_noise_vec;
        end
        
        clear N_cd N_c;
        end % if opt.maxevals_c >0
        
        hypersamples(hypersample_ind).hyperparameters(w0_inds) = ...
            mean(log_w_0_mat, 1);
        hypersamples(hypersample_ind).hyperparameters(noise_ind) = ...
            mean(log_noise_mat(:));
        hypersamples(hypersample_ind).X_c = X_c;
        hypersamples(hypersample_ind).log_tw_c = log_tw_c;
        
        fprintf('\n')
        
    end
        
    quad_log_tw_c_scales = nan(1, num_dims);
    quad_X_c_scales = nan(1, num_dims);

    a_hps_compcell = cat(1,a_hps_cell{:});
    logL_compcell = cat(1,logL_cell{:});
    
    % We compile the data from all those likelihood evaluations over
    % different hypersamples, active centres, and dimensions, and use it to
    % estimate the scales of variation of the likelihood with respect to
    % changes in X_c(d), w_c(d), w_0(d) and the noise. We'll also estimate
    % the output scale and noise for the likelihood function. We still have
    % to estimate the scale of variation wrt the output scale.
    
    for d = 1:num_dims
        % a_hps_mat has columns corresponding to:
        % a_X_c, a_w_c, w0, noise.
        
        a_hps_mat = cat(1,a_hps_compcell{:,d});
        logL_mat = cat(1,logL_compcell{:,d});
        
        sorted_logL_mat = sort(logL_mat);
              
        top_inds = logL_mat >= sorted_logL_mat(round(0.9*end));
        a_hps_mat = a_hps_mat(top_inds,:);
        logL_mat = logL_mat(top_inds,:);
       
        %a_hps_mat(:,4) = max(a_hps_mat(:,4), -100);

        [dummy, a_quad_input_scales] = ...
            hp_heuristics(a_hps_mat,logL_mat,100);
        %a_quad_input_scales = 10 * a_quad_input_scales;

        quad_X_c_scales(d) = a_quad_input_scales(1);
        quad_log_tw_c_scales(d) = a_quad_input_scales(2);      
    end
    
    
    
  
    fprintf('\n');
    
end

gp.hypersamples = hypersamples;
gp.X_data = X_data;
gp.y_data = y_data;
gp.active_hp_inds = full_active_inds;
gp.w0_inds = w0_inds;

[max_logL, max_ind] = max([gp.hypersamples.logL]);

fprintf('\n Initial best log-likelihood: \t%g',init_max_logL);
fprintf('\n Final best log-likelihood: \t%g',max_logL);
if opt.verbose
    fprintf(', for ')
    disp_spgp_hps(gp, max_ind, 'no_logL');
else
    fprintf('\n');
end



quad_gp.quad_noise_sd = quad_noise_sd;
quad_gp.quad_input_scales = quad_input_scales;
quad_gp.quad_output_scale = quad_output_scale;

warning('on','revise_spgp:X_c_problems')
fprintf('Completed retraining of GP in %g seconds\n', cputime-start_time)
fprintf('\n');

function [a_X_c_d, a_log_tw_c_d, log_w_0_d, log_noise...
                    a_hps_mat, logL_mat] = move_c(d, ...
                gp, quad_X_c_scales, quad_log_tw_c_scales, quad_input_scales, ...
                X_data, y_data, ...
                a_X_c_inds, a_w_c_inds, w0_inds, noise_ind, opt)
% we always move the first element of X_c and w_c. Note the scales of
% variation of the likelihood in X_c should be approximately input_scales,
% and approximately quad_input_scales for a variation in w_c.

w0_ind = w0_inds(d);

input_scales = [...
    quad_X_c_scales(d), ...
    quad_log_tw_c_scales(d), ...
    quad_input_scales([w0_ind, noise_ind])];
active_hp_inds = [...
    a_X_c_inds(1,d), ...
    a_w_c_inds(1,d), w0_ind, noise_ind];

a_X_c_d = gp.hypersamples.X_c(1,d);
log_a_tw_c_d = gp.hypersamples.log_tw_c(1,d);
log_w0_d = gp.hypersamples.hyperparameters(w0_ind);
log_noise = gp.hypersamples.hyperparameters(noise_ind);

a_hps = [...
    a_X_c_d, ...
    log_a_tw_c_d, log_w0_d, log_noise];

a_X_c_d_pos = 1;
a_w_c_d_pos = max(a_X_c_d_pos) + (1);
w_0_d_pos = max(a_w_c_d_pos) + 1;
noise_pos = max(w_0_d_pos) + 1;

if opt.verbose && opt.plots
    scrsz = [1 1 2000, 1000];%get(0,'ScreenSize');
    num_plots = length(active_hp_inds);
    for a = 1:num_plots
        figure(a);
        set(gca, 'TickDir', 'out')
        set(gca, 'Box', 'off', 'FontSize', 10); 
        set(gcf, 'color', 'white'); 
        set(gca, 'YGrid', 'off');
        set(gcf, 'position', [1+(a-1)*scrsz(3)/num_plots, scrsz(4)/2, ...
            scrsz(3)/num_plots, scrsz(4)/2]); 
        clf; 
        hold on;
        switch a
            case a_X_c_d_pos
                titley = ['X_c(',num2str(d),')'];
            case a_w_c_d_pos
                titley = ['w_c(',num2str(d),')'];
            case w_0_d_pos
                titley = ['w_0(',num2str(d),')'];
            case noise_pos
                 titley = ['noise sd'];
        end
        title(titley);
    end
end

flag = false;
i = 0;
a_hps_mat = nan(opt.maxevals_c,max(noise_pos));
logL_mat = nan(opt.maxevals_c,1);

broken = false;

while (~flag || ceil(opt.maxevals_c/5) > i) && i < opt.maxevals_c-1
    i = i+1;
    
    try
        gp = ...
            revise_spgp(X_data, y_data, gp, 'overwrite', [], active_hp_inds);
    catch
        broken = true;
        i = i - 1;
        break;
    end
    
    
    logL = gp.hypersamples.logL;
    
    a_hps_mat(i,:) = a_hps;
    logL_mat(i) = logL;
    
    if opt.verbose && opt.plots
        for a = 1:length(active_hp_inds)
            figure(a)
            x = a_hps(a);
            plot(x, logL, 'k.','MarkerSize',10);
            g = gp.hypersamples.glogL(a);
            scale = input_scales(a);

            line([x-scale,x+scale],...
                [logL-g*scale,logL+g*scale],...
                'Color',[0 0 0],'LineWidth',1.5);
        end
    end
  
    
    if i>1 && logL_mat(i) < backup_logL

        
         %[~,input_scales] = hp_heuristics(a_hps_mat(1:i,:),logL_mat(1:i,:),10);
        
        % the input scale which predicted the largest increase in logL is
        % likely wrong

         
         dist_moved = (a_hps - backup_a_hps).*a_grad_logL';
        [dummy,max_ind] = max(dist_moved);

        input_scales(max_ind) = 0.5*input_scales(max_ind);
         
        
        a_hps = backup_a_hps;
    else
        backup_a_hps = a_hps;
        backup_logL = logL;
        a_grad_logL = gp.hypersamples.glogL;
    end
    

    [a_hps, flag] = simple_zoom_pt(a_hps, a_grad_logL, ...
                            input_scales, 'maximise');
                        
    gp.hypersamples.X_c(1,d) = a_hps(a_X_c_d_pos); 
    gp.hypersamples.log_tw_c(1,d) = a_hps(a_w_c_d_pos);             
    gp.hypersamples.hyperparameters(w0_ind) = a_hps(w_0_d_pos);
    gp.hypersamples.hyperparameters(noise_ind) = a_hps(noise_pos);
    
end

if ~broken
    try
    gp = revise_spgp(X_data, y_data, gp, 'overwrite');
    logL = gp.hypersamples.logL;
    
    i = i+1;

    a_hps_mat(i,:) = a_hps;
    logL_mat(i) = logL;
    catch
    end
end
a_hps_mat = a_hps_mat(1:i,:);
logL_mat = logL_mat(1:i,:);
    

[max_logL,max_ind] = max(logL_mat);
a_hps = a_hps_mat(max_ind,:);

a_X_c_d = a_hps(a_X_c_d_pos); 
a_log_tw_c_d = a_hps(a_w_c_d_pos);   
log_w_0_d = a_hps(w_0_d_pos);
log_noise = a_hps(noise_pos);

if opt.verbose
    fprintf('LogL: %g -> %g',logL_mat(1),max_logL)
    fprintf(', tw_c: %g -> %g, scale: %g',...
        exp(a_hps_mat(1,a_w_c_d_pos(1))), ...
        exp(a_log_tw_c_d), ...
        input_scales(a_w_c_d_pos))
    fprintf(', X_c: %g -> %g, scale: %g\n',a_hps_mat(1,a_X_c_d_pos), a_X_c_d, input_scales(a_X_c_d_pos))
    

    %disp_spgp_hps(gp)
else
    fprintf('.')
end

if opt.verbose && opt.plots
    %title('Optimising c')
    
    
%     figure(5)
%     hold on
%     
%     wcs = exp(a_hps_mat(:,2));
%     w0s = logistic(a_hps_mat(:,3), w_max_mat);
%     plot(wcs,'.r')
%     plot(w0s,'.b')
%     figure(6)
%     plot3(wcs, w0s, logL_mat, '+-');
%     xlabel('w_c');
%     ylabel('w_0');
%     zlabel('log L');

1;
    
end


function [hypersample, a_hps_mat, logL_mat] = move_hypersample(...
    hypersample, gp, quad_input_scales, active_hp_inds, X_data, y_data, opt)

gp.hypersamples = hypersample;
a_quad_input_scales = quad_input_scales(active_hp_inds);

flag = false;
i = 0;
a_hps_mat = nan(opt.maxevals_hs,length(active_hp_inds));
logL_mat = nan(opt.maxevals_hs,1);

if opt.verbose && opt.plots
    for a = 1:length(active_hp_inds)
        figure(a);clf; hold on;
        title(['Optimising hyperperparameter ',num2str(a)])
    end
end

broken = false;

while (~flag || ceil(opt.maxevals_hs/5) > i) && i < opt.maxevals_hs-1
    i = i+1;
    
    try
        gp = ...
            revise_spgp(X_data, y_data, gp, 'overwrite', [], active_hp_inds);
    catch
        broken = true;
        i = i - 1;
        break;
    end
    
    logL = gp.hypersamples.logL;
%     if opt.verbose
%         fprintf('%g,',logL)
%     end
    a_hs=gp.hypersamples.hyperparameters(active_hp_inds);
    
    a_hps_mat(i,:) = a_hs;
    logL_mat(i) = logL;
    
    if opt.verbose && opt.plots
        for a = 1:length(active_hp_inds)
            figure(a)
            x = a_hs(a);
            plot(x, logL, '.');

            g = gp.hypersamples.glogL(a);
            scale = a_quad_input_scales(a);

            line([x-scale,x+scale],...
                [logL-g*scale,logL+g*scale],...
                'Color',[0 0 0],'LineWidth',1.5);
        end
    end
    
    if i>1 && logL_mat(i) < backup_logL
        % the input scale which predicted the largest increase in logL is
        % likely wrong
        
        dist_moved = (a_hs - backup_a_hs).*a_grad_logL';
        [dummy,max_ind] = max(dist_moved);

        a_quad_input_scales(max_ind) = 0.5*a_quad_input_scales(max_ind);
        
%         [~,a_quad_input_scales] = ...
%             hp_heuristics(a_hps_mat(1:i,:),logL_mat(1:i,:),10);
%         
        a_hs = backup_a_hs;
    else
        backup_logL = logL;
        backup_a_hs = a_hs;
        a_grad_logL = gp.hypersamples.glogL;
    end
    

    [a_hs, flag] = simple_zoom_pt(a_hs, a_grad_logL, ...
                            a_quad_input_scales, 'maximise');
    gp.hypersamples.hyperparameters(active_hp_inds) = a_hs;
    
end

if ~broken
    try

    
    gp = revise_spgp(X_data, y_data, gp, 'overwrite');
    logL = gp.hypersamples.logL;
    a_hs = gp.hypersamples.hyperparameters(active_hp_inds);

    i = i+1;
    
    a_hps_mat(i,:) = a_hs;
    logL_mat(i) = logL;
    catch
    end
end

a_hps_mat = a_hps_mat(1:i,:);
logL_mat = logL_mat(1:i,:);

[max_logL,max_ind] = max(logL_mat);
gp.hypersamples.hyperparameters(active_hp_inds) = a_hps_mat(max_ind,:);
gp = revise_spgp(X_data, y_data, gp, 'overwrite');
hypersample = gp.hypersamples;

% not_nan = all(~isnan([a_hps_mat,logL_mat]),2);
% 
% [quad_noise_sd, a_quad_input_scales] = ...
%     hp_heuristics(a_hps_mat(not_nan,:), logL_mat(not_nan), 10);
% quad_input_scales(active_hp_inds) = a_quad_input_scales;

if opt.verbose
fprintf('LogL: %g -> %g',logL_mat(1), max_logL)
else
    fprintf('.')
end
if opt.verbose && opt.plots
    %keyboard;
end