
function [minimum, minimum_location, X_data, y_data, gp, quad_gp] = ...
  gpgo(fn, X_0, lower_bound, upper_bound, opt)
% [minimum, minimum_location, X_data, y_data, gp, quad_gp] = ...
%   gpgo(fn, X_0, lower_bound, upper_bound, opt)
% minimise function fn by sequentially greedily selecting the most valuable
% observation according to a GP fit to fn.
%
% below are the fields permitted to opt and their default values
%
% allowed number of function evals to take:
%   'function_evaluations', 100 * num_dims, ...
% total time allowed to gpgo:
%   'total_time', 3600*24*7, ...
% stop if target is reached:
%   'target', -inf, ...
% the number of steps to lookahead (only 1 is implemented)
%   'lookahead_steps', 1, ...
% user observations of the gradient of the function
%   'derivative_observations', nargout(fn)>1, ...
% which minimiser to use for the expected loss surface
%   'exp_loss_minimiser',@fast_exp_loss_min,...
% how many evaluations of the expected loss surface that minimiser is
% allowed
%   'exp_loss_evals', 100 * num_dims,...
% covariance function used by gpgo
%   'cov_fn','matern', ...
% mean function used by gpgo
%   'mean_fn', 'constant', ...
% number of hyperparameter samples (weighted using bayesian quadrature) to
% use
%   'num_hypersamples', 10 * num_dims, ...
% retrain the gp after every retrain_period evaluations of the function
%   'retrain_period', 10 * num_dims, ...
% number of likelihood evaluations to allow training process
%   'train_evals', 10 * num_dims, ...
% make plots as the algorithm proceeds
%   'plots', false, ...
% if ~save_str, save after each evaluation to a file with name save_str
%   'save_str', false, ...
% print diagnostic information to the command window
%   'verbose', false);


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

default_opt = struct( ...
    'function_evaluations', 100 * num_dims, ...
    'total_time', 3600*24*7, ...
    'target', -inf, ...
    'lookahead_steps', 1, ...
    'derivative_observations', nargout(fn)>1, ...
    'exp_loss_minimiser',@fast_exp_loss_min,...
    'exp_loss_evals', 50 * num_dims,...
    'cov_fn','matern', ...
    'mean_fn', 'constant', ...
    'num_hypersamples', 10 * num_dims, ...
    'retrain_period', 10  * num_dims, ...
    'train_evals', 100 * num_dims, ...
    'plots', false, ...
    'save_str', false, ...
    'verbose', false, ...
    'pool_open', false ...
);


if isfield(opt,'total_time')
    % we probably want to use total_time rather than function_evaluations
    % as our stopping criterion
    opt.function_evaluations = 3000;
end

names = fieldnames(default_opt);
for i = 1:length(names);
    name = names{i};
    if (~isfield(opt, name))
      opt.(name) = default_opt.(name);
    end
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
input_scales = (upper_bound-lower_bound)/10;
if any(input_scales==0)
    error('lower bound is equal to upper bound');
end
gp = set_gp(opt.cov_fn, opt.mean_fn, gp, input_scales, [], ...
    opt.num_hypersamples);

gp = hyperparams(gp);
num_hypersamples = numel(gp.hypersamples);
lastHyperSamplesMoved = 1:num_hypersamples;

gp.grad_hyperparams = false;
weights_mat = bq_params(gp);

errors = struct();

if opt.derivative_observations

    hps_struct = set_hps_struct(gp);
    % need to define this handle or else infinite recursion results
    gp.non_deriv_cov_fn = gp.covfn;
    gp.covfn = @(flag) derivativise(gp.non_deriv_cov_fn,flag);
    gp.meanfn = @(flag) wderiv_mean_fn(hps_struct,flag);
end

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

    %if opt.verbose
        fprintf('Minimum:\t%g\n',min_so_far);
    %else
%         fprintf('.');
%     end


    if rem(evaluation,opt.retrain_period) ~= 0
        % for hypersamples that haven't been moved, update
        gp = revise_gp(X_data_so_far, y_data_so_far, ...
            gp, 'update', eval_inds, ...
            setdiff(1:num_hypersamples,lastHyperSamplesMoved));

        % for hypersamples that have been moved, overwrite
        gp = revise_gp(X_data_so_far, y_data_so_far, ...
            gp, 'overwrite', [], ...
            lastHyperSamplesMoved);

        lastHyperSamplesMoved = [];
    else
        % retrain gp
        [gp, quad_gp] = ...
            train_gpgo(gp, X_data_so_far, y_data_so_far, opt);

        weights_mat = bq_params(gp, quad_gp);

        num_hypersamples = numel(gp.hypersamples);
        lastHyperSamplesMoved = 1:num_hypersamples;
    end
%     hs_weights = zeros(1,num_hypersamples);
%     [maX_logL,maX_logL_ind] = max([gp.hypersamples(:).logL]);
%     hs_weights(maX_logL_ind) = 1;

    hs_weights = weights(gp, weights_mat);
    if any(isnan(hs_weights))
        % something numerical has gone wrong in BQ, try to recover by just
        % using the hypersmaple with maximum likelihood
        new_hs_weights = zeros(size(hs_weights));
        [unused_maximum, max_weighted_ind] = max([gp.hypersamples(:).logL]);
        new_hs_weights(max_weighted_ind) = 1;
        hs_weights = new_hs_weights;
    else
        [max_weights, max_weighted_ind] = max(hs_weights);
    end
    gp.max_weighted_hypersample_ind = max_weighted_ind;

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

    x = opt.exp_loss_minimiser(exp_loss, ...
                lower_bound, upper_bound, opt.exp_loss_evals, ...
                X_data_so_far, min_ind, opt);

    if opt.plots && num_dims == 1
        clf

        X_star = linspace(lower_bound,upper_bound,100)';
        for i = 1:100
            [m(i), vars(i)] = posterior_gp(X_star(i), gp, max_weighted_ind,...
                {'var_not_cov','jitter_corrected'});
        end
        params.width = 25;
        params.height = 35;
        gp_plot(X_star, m, sqrt(vars), X_data_so_far, y_data_so_far,...
            [],[],params);


    end
    if opt.save_str
        save(opt.save_str);
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

if opt.save_str
    save(opt.save_str);
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

function X_min = exp_loss_direct(exp_loss, ...
                lower_bound, upper_bound, exp_loss_evals,...
                X_data, min_ind, opt)
% find the position at which fn exp_loss is minimised.

Problem.f = @(x) exp_loss(x');

opts.maxevals = exp_loss_evals;
opts.showits = 0;
bounds = [lower_bound; upper_bound]';

[exp_loss_min, X_min] = Direct(Problem, bounds, opts);
X_min = X_min';

function X_min = fast_exp_loss_min(exp_loss, ...
                lower_bound, upper_bound, exp_loss_evals, ...
                X_data, y_min_ind, opt)
% find the position at which fn exp_loss is minimised.

bounds = [lower_bound;upper_bound];

if opt.derivative_observations
    plain_obs = find(X_data(:,end) == 0);
    X_data = X_data(plain_obs,1:end-1);
    exp_loss = @(x) exp_loss([x,0]);
    y_min_ind = find(plain_obs==y_min_ind);
end

num_data = size(X_data,1);
num_dims = size(X_data,2);

start_pt_inds = round(linspace(1, num_data, 5));
start_pt_inds = unique([y_min_ind, start_pt_inds(2:end)]);
num_start_pts = length(start_pt_inds);
start_pts = X_data(start_pt_inds,:);

num_line_pts = floor(0.5*exp_loss_evals/num_start_pts);

best_X = nan(num_start_pts, num_dims);
best_loss = nan(num_start_pts,1);

if opt.plots
    switch num_dims
        case 1
            x = linspace(lower_bound,upper_bound, 100);
            f = x;
            for i = 1:length(x)
                f(i) = exp_loss(x(i));
            end

            plot(x,f,'r');
            hold on
        case 2

            num = 30;

            x1 = linspace(lower_bound(1),upper_bound(1), num)';
            x2 = linspace(lower_bound(2),upper_bound(2), num)';
            x = allcombs([x1,x2]);
            f = nan(length(x),1);
            for i = 1:length(x)
                f(i) = exp_loss(x(i,:));
            end

            X2 = repmat(x2,1,num);
            X1 = repmat(x1',num,1);
            F = reshape(f,num,num);

            max_y = max(f+1);

            clf
            contourf(X1,X2,F);
            colorbar
            hold on
            plot3( ...
                X_data(:,1), ...
                X_data(:,2), ...
                max_y*ones(num_data,1), ...
                'w+', ...
                'LineWidth', 5, ...
                'MarkerSize', 10 ...
            )
            plot3( ...
                X_data(:,1), ...
                X_data(:,2), ...
                max_y*ones(num_data,1), ...
                'k+', ...
                'LineWidth', 3, ...
                'MarkerSize', 9 ...
            )

    end
end


% % assume the input scales for the expected loss surface are proportional to
% % those of the objective fn.
% [unused_max_logL, best_ind] = max([gp.hypersamples(:).logL]);
% input_scales = exp(...
%     gp.hypersamples(best_ind).hyperparameters(gp.input_scale_inds));

% We find the best expected loss local to a small number of starting points

x = nan(num_start_pts, num_dims);
f = nan(num_start_pts, 1);

for start_pt_ind = 1:num_start_pts
    % number of data larger than number of hypersamples, so we use parfor here

    start_pt = start_pts(start_pt_ind,:);

    input_scales = 0.5*max(eps,min(bsxfun(@(x,y) abs(x-y), start_pt, ...
        X_data(setdiff(1:end,start_pt_inds(start_pt_ind)),:)),[],1));

    [unused_f, g] = exp_loss(start_pt);

    zoomed = ...
        simple_zoom_pt(start_pt, g, input_scales, 'minimise');

    x(start_pt_ind,:) = cap(zoomed, lower_bound, upper_bound);
    f(start_pt_ind) = exp_loss(x(start_pt_ind,:));
end

% Then, for each starting point, we perform a line search in the direction
% given by the difference between the starting point and its local optimum.
% More accurately, for a number of points along that line, we perform
% another local optimisation.

for start_pt_ind = 1:num_start_pts

    start_pt = start_pts(start_pt_ind,:);

    input_scales = 0.5*max(eps,min(bsxfun(@(x,y) abs(x-y), start_pt, ...
        X_data(setdiff(1:end,start_pt_inds(start_pt_ind)),:)),[],1));

    best_X_line = nan(num_line_pts, num_dims);
    best_loss_line = nan(num_line_pts,1);

    best_X_line(1,:) = x(start_pt_ind,:);
    best_loss_line(1) = f(start_pt_ind,:);

    direction = x(start_pt_ind,:) - start_pt;
    if norm(direction) == 0
        % move towards a corner. The corner is chosen by rotating through
        % them all using a binary conversion.

        bin_vec = de2bi(start_pt_inds(start_pt_ind))+1;

        sel_vec = ones(1,num_dims);
        num = min(length(sel_vec),length(bin_vec));
        sel_vec(1:num) = bin_vec(1:num);

        direction =...
            bounds(sub2ind([2,num_dims],sel_vec,1:num_dims)) - start_pt;
        if norm(direction) == 0
            direction = mean([lower_bound;upper_bound]) - start_pt;
        end
    end
    unzeroed_direction = direction;
    unzeroed_direction(abs(direction)<eps) = eps;

    ups = (upper_bound - start_pt)./unzeroed_direction;
    downs = (lower_bound - start_pt)./unzeroed_direction;
    bnds = sort([ups;downs]);
    min_bnd = max(bnds(1,:));
    max_bnd = min(bnds(2,:));

    X_line = @(line_pt) start_pt + line_pt*direction;

    if opt.plots

        switch num_dims
            case 1

            case 2
                coords = [X_line(min_bnd); X_line(max_bnd)];
                line(coords(:,1), coords(:,2), [max_y,max_y], 'Color','w')
        end
    end

    line_pts = linspace(min_bnd, max_bnd, num_line_pts-1);

    parfor line_pt_ind = 2:num_line_pts
        line_pt = line_pts(line_pt_ind-1);
        X_line_pt = X_line(line_pt);

        [original_loss, g] = exp_loss(X_line_pt);
        zoomed = ...
            simple_zoom_pt(X_line_pt, g, input_scales, 'minimise');
        zoomed = cap(zoomed, lower_bound, upper_bound);

        zoomed_loss = exp_loss(zoomed);

        if original_loss < zoomed_loss
            best_X_line(line_pt_ind,:) = X_line_pt;
            best_loss_line(line_pt_ind) = original_loss;
        else
            best_X_line(line_pt_ind,:) = zoomed;
            best_loss_line(line_pt_ind) = zoomed_loss;
        end

%         if opt.plots
%             switch num_dims
%                 case 1
%                     plot(zoomed,best_loss_line(line_pt_ind),'ro','MarkerSize',6)
%                 case 2
%                     plot3(X_line_pt(1),X_line_pt(2),max_y,'w.','MarkerSize',10)
%                     plot3(zoomed(1),zoomed(2),max_y,'w.','MarkerSize',14)
%                     plot3(zoomed(1),zoomed(2),max_y,'w.','MarkerSize',6)
%             end
%         end
    end

    [min_best_loss_line, min_ind_line] = min(best_loss_line);

    best_X(start_pt_ind,:) = best_X_line(min_ind_line,:);
    best_loss(start_pt_ind,:) = min_best_loss_line;
end

[min_best_loss, min_ind] = min(best_loss);
X_min = best_X(min_ind,:);

if opt.plots
    switch num_dims
        case 1
            plot(X_min,min_best_loss,'r+','MarkerSize',10)
            refresh
            drawnow
        case 2
            plot3(X_min(1),X_min(2),max_y,'w+','LineWidth',5,'MarkerSize',10)
            refresh
            drawnow
    end
end


function [f,g,H] = weighted_neg_val(rho,XStar,gp,varargin)

NStar=size(XStar,1);
f=zeros(NStar,1);
g=0;
H=0;
switch nargout
    case {0,1}
        for sample=1:numel(gp.hypersamples)
                    fi=negval(XStar,gp,sample,varargin{:});
                    f=f+rho(sample)*fi;
        end
    case 2
        for sample=1:numel(gp.hypersamples)
                    [fi,gi]=negval(XStar,gp,sample,varargin{:});
                    f=f+rho(sample)*fi;
                    g=g+rho(sample)*gi;
        end
    case 3
        for sample=1:numel(gp.hypersamples)
                    [fi,gi,Hi]=negval(XStar,gp,sample,varargin{:});
                    f=f+rho(sample)*fi;
                    g=g+rho(sample)*gi;
                    H=H+rho(sample)*Hi;
        end
end

function x = cap(x, lower_bound, upper_bound)
x = min(max(x, lower_bound), upper_bound);
