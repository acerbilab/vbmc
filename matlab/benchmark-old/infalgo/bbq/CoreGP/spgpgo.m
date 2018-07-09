
function [minimum, minimum_location, X_data, y_data, gp, quad_gp] = ...
  spgpgo(fn, X_0, lower_bound, upper_bound, opt, gp)
% [minimum, minimum_location, X_data, y_data, gp, quad_gp] = ...
%   gpgo(fn, X_0, lower_bound, upper_bound, opt, gp)
% minimise function fn by sequentially greedily selecting the most valuable
% observation according to a GP fit to fn. 
%
% default_opt = struct('function_evaluations', 100 * num_dims, ...
%                         'total_time', 3600*24*7, ...
%                         'target', -inf, ...
%                        'lookahead_steps', 1, ...
%                        'derivative_observations', nargout(fn)>1, ...
%                        'input_scales', 1,...
%                        'exp_loss_minimiser',@fast_exp_loss_min,...
%                        'exp_loss_evals', 100 * num_dims,...
%                        'cov_fn','matern', ...
%                        'mean_fn', 'constant', ...
%                        'num_hypersamples', 10 * num_dims, ...
%                        'retrain_period', 10 * num_dims, ...
%                        'train_evals', 100 * num_dims, ...
%                        'plots', false, ...
%                        'save_str', false, ...
%                         'verbose', false);

if size(X_0,1)~=1 && size(X_0,2)==1
    X_0 = X_0';
end
if size(lower_bound,1)~=1 && size(lower_bound,2)==1
    lower_bound = lower_bound';
end
if size(upper_bound,1)~=1 && size(upper_bound,2)==1
    upper_bound = upper_bound';
end

num_dims = length(X_0);

if nargin<5
    opt = struct();
end

default_opt = struct('function_evaluations', 1000 * num_dims, ...
                        'total_time', 3600*24*7, ...
                        'target', -inf, ...
                       'lookahead_steps', 1, ...
                       'derivative_observations', nargout(fn)>1, ...
                       'input_scales', 1,...
                       'exp_loss_minimiser',@fast_exp_loss_min,...
                       'exp_loss_evals', 50 * num_dims,...
                       'mean_fn', 'constant', ...
                       'num_hypersamples', 10 * num_dims, ...
                       'num_c', 500, ...
                       'num_retrains', 10, ...
                       'optim_time', 70 * num_dims, ...
                       'plots', false, ...
                       'save_str', false, ...
                        'verbose', false, ...
                        'pool_open', false);
           
time_limited = isfield(opt,'total_time');
                   
if time_limited
    % we probably want to use total_time rather than function_evaluations
    % as our stopping criterion
    opt.function_evaluations = 100000;
end

names = fieldnames(default_opt);
for i = 1:length(names);
    name = names{i};
    if (~isfield(opt, name))
      opt.(name) = default_opt.(name);
    end
end

% don't want to train right at the end
if time_limited
    
    num_possible_retrains = floor(opt.total_time/opt.optim_time);
    num_retrains = min(opt.num_retrains, num_possible_retrains);
    
    retrain_times = intlogspace(1, opt.total_time, ...
                                    num_retrains+2);
    delete_inds = find(diff(retrain_times) <= opt.optim_time);
    delete_inds = [delete_inds, 1];
    retrain_times(delete_inds) = [];                      
    retrain_times(end) = inf;
    
else
    retrain_inds = intlogspace(ceil(min(opt.num_c,opt.function_evaluations)/10), ...
                                    opt.function_evaluations, ...
                                    opt.num_retrains+1);
    retrain_inds(end) = inf;
end

if ~opt.pool_open
    matlabpool close force
    matlabpool open
end


if opt.derivative_observations
    % assumed that each evaluation of the function yield both the
    % appropriate value of the objective function along with the gradient
    % at that point, as per [y,g] = fn(x). We store the gradient by adding
    % an additional num_dimension to the end of x, containing an integer
    % indicating the element of the gradient that the corresponding element
    % of y contains. If the additional num_dimsension is 0, that
    % corresponds to an observation of the objective observation.
    total_data = opt.function_evaluations*(num_dims+1);
    X_data = nan(total_data, num_dims+1);
    y_data = nan(total_data, 1);
    
    gp.sqd_diffs_cov = false;
else
    X_data = nan(opt.function_evaluations, num_dims);
    y_data = nan(opt.function_evaluations, 1);
end

x = X_0;



% initialise gp
gp.hyperparams(1)=struct('name','logNoiseSD',...
        'priorMean',log(eps),...
        'priorSD',eps,...
        'NSamples',1,...
        'type','inactive');
gp.logNoiseSDPos = 1;
input_scales = (upper_bound-lower_bound)/10;
if all(input_scales==0)
    error('lower bound is equal to upper bound');
end
gp = set_spgp(opt.mean_fn, gp, input_scales, [], ...
    opt.num_c, ...
    opt.num_hypersamples);



just_moved_hypersamples = true;

gp.grad_hyperparams = false;
weights_mat = bq_params(gp);

errors = struct();

% if opt.derivative_observations
%     
%     hps_struct = set_hps_struct(gp);
%     gp.meanfn = @(flag) wderiv_mean_fn(hps_struct,flag);
% end

start_time = cputime;
evaluation = 0;
while evaluation < opt.function_evaluations && ...
        (cputime - start_time) < opt.total_time
    evaluation = evaluation+1;
    
    if opt.derivative_observations
        try
            [y,g] = fn(x);
            if size(g,1)==1;
                g=g';
            end
        catch ME
            y = max([y_data;inf]);
            g = zeros(num_dims,1);
            error_msg = getReport(ME);
            disp(error_msg);
            errors(numel(errors)+1).msg = error_msg;
        end
        
        eval_inds = (evaluation-1)*(num_dims+1)+(1:num_dims+1);
        X_data(eval_inds, :) = [repmat(x,num_dims+1,1),(0:num_dims)'];
        y_data(eval_inds) = [y;g];
        
        all_eval_inds = 1:max(eval_inds);
        X_data_so_far = X_data(all_eval_inds, :);
        y_data_so_far = y_data(all_eval_inds);

        plain_obs = find(X_data_so_far(:,end) == 0);
    
        [min_so_far, min_ind] = min(y_data_so_far(plain_obs,:));
        min_ind = plain_obs(min_ind);
        
    else
        try
            y = fn(x);
        catch ME
            y = max([y_data;inf]);
            error_msg = getReport(ME);
            disp(error_msg);
            errors(numel(errors)+1).msg = error_msg;
        end
        
        eval_inds = evaluation;
        X_data(eval_inds, :) = x;
        y_data(eval_inds) = y;
        
        all_eval_inds = 1:max(eval_inds);
        X_data_so_far = X_data(all_eval_inds, :);
        y_data_so_far = y_data(all_eval_inds);

        [min_so_far, min_ind] = min(y_data_so_far);
        
    end
    
    if y<opt.target
        % a sufficiently low y has been found.
        break
    end
    
    fprintf('Minimum:\t%g\n',min_so_far);
    
    if time_limited
        retrain_now = (cputime - start_time) >= retrain_times(1);
    else
        retrain_now = evaluation >= retrain_inds(1); 
    end
    
    still_adding_centres = evaluation < opt.num_c;
    if still_adding_centres && ~retrain_now;
        % keep adding in those centres at observations
        
        gp = set_spgp(opt.mean_fn, [], X_data_so_far, y_data_so_far, ...
            opt.num_c, ...
            opt.num_hypersamples);
        
        just_moved_hypersamples = true;
        
    end
    
    
    if ~retrain_now 
        if ~just_moved_hypersamples
            % for hypersamples that haven't been moved, update
            gp = revise_spgp(X_data_so_far, y_data_so_far, ...
                gp, 'update', eval_inds);
        else
            % for hypersamples that have been moved, overwrite
            gp = revise_spgp(X_data_so_far, y_data_so_far, ...
                gp, 'overwrite');
        end
        
        just_moved_hypersamples = false;
        
        if still_adding_centres
            % the new centres added will probably have drastically changed
            % the likelihood landscape. bq_params contains an hp_heuristics
            % call to estimate the quadrature hyperparameters
            weights_mat = bq_params(gp);
        end
        
    else
        % retrain gp
        [gp, quad_gp] = ...
            train_spgp([], X_data_so_far, y_data_so_far, opt);
        
        weights_mat = bq_params(gp, quad_gp);
        
        just_moved_hypersamples = true;
        
        if time_limited
            retrain_times(1) = [];
        else
            retrain_inds(1) = [];
        end
    end
    
    hs_weights = weights(gp, weights_mat);
     
    if (evaluation == opt.function_evaluations)
        continue
    elseif (evaluation == 1) % we should just go to a corner
        x = lower_bound;
        continue
    end
       
  	if (opt.lookahead_steps == 1)
			exp_loss = @(XStar) weighted_neg_val(hs_weights,XStar,...
                       gp,min_so_far);
		else
			exp_loss = @(XStar) multi_step_negvalue(hs_weights,XStar,...
                       gp,min_so_far);
    end
    
    [noise, lambda, w_0, X_c, w_c]  = ...
        disp_spgp_hps(gp);
    x = opt.exp_loss_minimiser(exp_loss, ...
                lower_bound, upper_bound, opt.exp_loss_evals, ...
                X_data_so_far, w_0, min_ind, opt);
	
    if opt.plots
        [max_weights,max_weighted_ind] = max(hs_weights);
        switch num_dims


            case 1
                clf

               
                X_star = linspace(lower_bound,upper_bound,100)';
                for i = 1:100
                    [m(i), vars(i)] = posterior_spgp(X_star(i), gp, max_weighted_ind,...
                    {'var_not_cov'});
                end
                params.width = 25;
                params.height = 35;
                gp_plot(X_star, m, sqrt(vars), X_data_so_far, y_data_so_far,...
                    [],[],params);
            case 2
        

                num = 100;

                x1 = linspace(lower_bound(1),upper_bound(1), num)';
                x2 = linspace(lower_bound(2),upper_bound(2), num)';
                xxx = allcombs([x1,x2]);
                
                params.print = false;
                [f{1}, f{2}] = predict_spgp(xxx, gp, weights_mat, params);
                
                X2 = repmat(x2,1,num);
                X1 = repmat(x1',num,1);
                
                X_c = gp.hypersamples(max_weighted_ind).X_c; 
                
                names = {'mean','sd'};
                for i = 1:2
                    F = reshape(f{i},num,num);

                    max_y = max(f{i}+1);

                    figure(i)
                    clf

                    contourf(X1,X2,F);
                    colorbar
                    hold on
                    title(names{i});

                    plot3(X_c(:,1),X_c(:,2),...
                        max_y*ones(size(X_c,1),1),...
                        'wo','LineWidth',5,'MarkerSize',10)
                    
                    
                    plot3(X_data_so_far(:,1),X_data_so_far(:,2),...
                        max_y*ones(evaluation,1),...
                        'w+','LineWidth',5,'MarkerSize',10)
                    plot3(X_data_so_far(:,1),X_data_so_far(:,2),...
                        max_y*ones(evaluation,1),...
                        'k+','LineWidth',3,'MarkerSize',9)

                end
                refresh
                drawnow
        end

    end
    if opt.save_str
        %save(opt.save_str,'-struct','gp');
        save(opt.save_str, 'X_data_so_far', 'y_data_so_far',...
            'noise', 'lambda', 'w_0', 'X_c', 'w_c', ...
            'errors')
    end
end

% end of optimisation

X_data = X_data_so_far;
y_data = y_data_so_far;

if opt.derivative_observations
    plain_obs = find(X_data(:,end) == 0);
    
    [minimum, min_ind] = min(y_data(plain_obs,:));
    min_ind = plain_obs(min_ind);
else
    [minimum, min_ind] = min(y_data);
end

minimum_location = X_data(min_ind,:);

if numel(errors)>1
    gp.errors = errors(2:end);
end

if nargout>5
    r_X_data = vertcat(gp.hypersamples.hyperparameters);
    r_y_data = vertcat(gp.hypersamples.logL);

    [quad_noise_sd, quad_input_scales, quad_output_scale] = ...
        hp_heuristics(r_X_data, r_y_data, 10);

    quad_gp.quad_noise_sd = quad_noise_sd;
    quad_gp.quad_input_scales = quad_input_scales;
    quad_gp.quad_output_scale = quad_output_scale;
end

if ~opt.pool_open
    matlabpool close
end

function X_min = exp_loss_direct(exp_loss, ...
                lower_bound, upper_bound, exp_loss_evals,...
                X_data, w_0, min_ind, opt)
% find the position at which fn exp_loss is minimised.

Problem.f = @(x) exp_loss(x');

opts.maxevals = exp_loss_evals;
opts.showits = 0;
bounds = [lower_bound; upper_bound]';

[exp_loss_min, X_min] = Direct(Problem, bounds, opts);
X_min = X_min';


function [f,g,H] = weighted_neg_val(rho,XStar,gp,varargin)

NStar=size(XStar,1);
f=zeros(NStar,1);
g=0;
H=0;
switch nargout 
    case {0,1}
        for sample=1:numel(gp.hypersamples)   
                    fi=spnegval(XStar,gp,sample,varargin{:});
                    f=f+rho(sample)*fi;
        end
    case 2
        for sample=1:numel(gp.hypersamples)   
                    [fi,gi]=spnegval(XStar,gp,sample,varargin{:});
                    f=f+rho(sample)*fi;
                    g=g+rho(sample)*gi;
        end
    case 3
        for sample=1:numel(gp.hypersamples)   
                    [fi,gi,Hi]=spnegval(XStar,gp,sample,varargin{:});
                    f=f+rho(sample)*fi;
                    g=g+rho(sample)*gi;
                    H=H+rho(sample)*Hi;
        end
end
