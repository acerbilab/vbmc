function [x_stars, m_stars, sd_stars, gp] = track_gp(x_d, y_d, ...
    gp, opt)
% [x_stars, m_stars, sd_stars, gp] = track_gp(x_d, y_d, ...
%    gp, opt, quad_gp)
% perform squential prediction with a pre-trained gp
% Set unspecified fields to default values. Retraining is performed
% intermittently.

if nargin < 4
    opt = struct();
end

[num_data, num_dims] = size(x_d);

default_opt = struct('lookahead', 1, ...
                     'steps_to_lookahead', 10, ...
                     'marg_hypers', nargin==5, ... % marginalise hyperparams using bayesian quadrature; otherwise perform maximum likelihood
                     'num_retrains', 10, ...
                     'train_gp_time', 50 * num_dims, ...
                     'parallel', true, ...
                     'train_gp_num_samples', 5 * num_dims, ...
                     'train_gp_print', false, ...
                     'print', 1);%'lengthscale');
opt = set_defaults( opt, default_opt );

% GP training options.
gp_train_opt.optim_time = opt.train_gp_time;
gp_train_opt.noiseless = true;
gp_train_opt.prior_mean = 0;
% print to screen diagnostic information about gp training
gp_train_opt.print = opt.train_gp_print;
% plot diagnostic information about gp training
gp_train_opt.plots = false;
gp_train_opt.parallel = opt.parallel;
gp_train_opt.num_hypersamples = opt.train_gp_num_samples;


% Specify iterations when we will retrain the GP on r.
retrain_inds = intlogspace(ceil(num_data/10), ...
                                num_data, ...
                                opt.num_retrains);
retrain_inds(end) = inf;


if ~isfield(gp, 'hypersamples')
% Initialises hypersamples
gp = hyperparams(gp);
end

if opt.print > 0 
    display('Beginning prediction')
    start_time = cputime;
end


num_predictions = (num_data - opt.lookahead);
num_samples = numel(gp.hypersamples);
num_stars = num_predictions * opt.steps_to_lookahead;



x_stars = nan(num_stars, num_dims);
m_stars = nan(num_stars, 1);
sd_stars = nan(num_stars, 1);


for i = 1:num_predictions
    
    x_up2i = x_d(1:i, :);
    y_up2i = y_d(1:i, :);
    
        
    % Retrain GP
    % ===========================   
    retrain_now = i >= retrain_inds(1);  % If we've passed the next retraining index.
    if i==1  % First iteration.

        % Set up GP without training it, because there's not enough data.
        gp_train_opt.optim_time = 0;
        [gp, quad_gp] = train_gp('sqdexp', 'constant', [], ...
                                     x_up2i, y_up2i, ...
                                     gp_train_opt);
                                 
        gp_train_opt.optim_time = opt.train_gp_time;
        
        if ~opt.marg_hypers
            [dummyVar,closestInd] = max([gp.hypersamples.logL]);
            hs_weights = zeros(num_samples, 1);
            hs_weights(closestInd) = 1;
        else
            weights_mat = bq_params(gp, quad_gp);
            hs_weights = weights(gp, weights_mat);
        end
        
    elseif retrain_now
        % Retrain gp.
        if opt.print == 1
            disp('retrain gp');
        end
        
        [gp, quad_gp] = train_gp('sqdexp', 'constant', gp, ...
                                     x_up2i, y_up2i, ...
                                     gp_train_opt);             
                                 
        retrain_inds(1) = [];   % Move to next retraining index. 
    else
        % for hypersamples that haven't been moved, update
        gp = revise_gp(x_up2i, y_up2i, ...
                         gp, 'update', i);
                    
    end
    
    x_star_0 = x_d(i, :);
    x_star_1 = x_d(i+opt.lookahead, :);
    x_star = linspace(x_star_0, x_star_1, opt.steps_to_lookahead)';
    
    
    m_star = 0;
    var_star = 0;
    for sample = 1:numel(gp.hypersamples)   
        [hs_m_star,hs_var_star] = posterior_gp(x_star,gp,sample,'var_not_cov');
        
        m_star = m_star + hs_weights(sample)*hs_m_star;
        var_star = var_star + hs_weights(sample)*(hs_var_star + hs_m_star.^2);
    end
    var_star = var_star - m_star.^2;
    sd_star = sqrt(var_star); 
    
    star_inds = (1:opt.steps_to_lookahead) + ...
                    (i-1) * opt.steps_to_lookahead;
    x_stars(star_inds) = x_star;
    m_stars(star_inds) = m_star;
    sd_stars(star_inds) = sd_star;
    
    if opt.print == 1
        if rem(i, 50) == 0
            fprintf('\n%g',i);
        else
            fprintf('.');
        end
    end
    
end

if opt.print==1
    fprintf('\n Prediction complete in %g second', cputime - start_time);
end