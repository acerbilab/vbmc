function invK_N_invK = bq_params(gp,quad_gp)
% places matrix invK_N_invK in gp, such that the weights over
% hypersamples are invK_N_invK*Lvec for Lvec the vector of likelihoods.
% Differs from bmcparams in that this can be used even where samples are
% not taken from a grid (but is probably slower) and also currently does
% not exploit likelihood derivatives.

if isfield(gp,'grad_hyperparams')
    grad_hyperparams = gp.grad_hyperparams;
else
    grad_hyperparams = isfield(gp.hypersamples,'glogL') && ...
        ~isempty(gp.hypersamples(1).glogL);
end
if ~isfield(gp, 'active_hp_inds')
    gp.active_hp_inds = 1:numel(gp.hyperparams);
end
active_hp_inds = gp.active_hp_inds;

hypersamples = remove_infs(cat(1,gp.hypersamples.hyperparameters));

num_hps = length(active_hp_inds);
num_data = size(hypersamples,1);

prior_means = remove_infs([gp.hyperparams.priorMean]);
prior_SDs = remove_infs([gp.hyperparams.priorSD]);

if nargin<2
    if size(hypersamples,1) == 1
        quad_input_scales = ones(1,max(active_hp_inds));
        quad_noise_sd = 0;
    elseif ~isfield(gp.hypersamples,'logL')
        quad_noise_sd = 0;

        r_XData = vertcat(gp.hypersamples.hyperparameters);
        quad_input_scales = max(1e-6,min(diff(sort(r_XData))));
    else
        r_XData = vertcat(gp.hypersamples.hyperparameters);
        r_yData = vertcat(gp.hypersamples.logL);

        [quad_noise_sd, quad_input_scales, quad_output_scale] = ...
            hp_heuristics(r_XData, r_yData, 10);

        lambda = quad_output_scale * ...
            prod(2*pi*quad_input_scales)^(0.25);
        quad_noise_sd = quad_noise_sd / lambda;

        quad_input_scales = 10*quad_input_scales;
    end
else
%     best_quad_hypersample_ind = max([likelihood_gp.hypersamples(:).logL]);
%     best_quad_hypersample = ...
%         likelihood_gp.hypersamples(best_quad_hypersample_ind).hyperparameters;
%
%     hps_struct = set_hps_struct(likelihood_gp);
%     quad_input_scales = exp(best_quad_hypersample(hps_struct.logInputScales));

    % quad_output_scale^2 = lambda^2 / sqrt(prod(2*pi*quad_input_scales
    lambda = quad_gp.quad_output_scale * ...
        prod(2*pi*quad_gp.quad_input_scales)^(0.25);

    quad_noise_sd = quad_gp.quad_noise_sd / lambda;
    quad_input_scales = quad_gp.quad_input_scales;
end
quad_input_scales = real(max(eps, quad_input_scales));

if ~grad_hyperparams
    K_q = ones(num_data);
    K_r = ones(num_data);
    N = ones(num_data);
else
    K_q = ones(num_data);
    K_r = ones(num_data*(num_hps+1));
    N = ones(num_data, num_data*(num_hps+1));
end

for active_ind = 1:length(active_hp_inds)

    hp = active_hp_inds(active_ind);

    width = quad_input_scales(hp);
    prior_SD_hp = prior_SDs(hp);
    prior_mean_hp = prior_means(hp);
    hypersamples_hp = hypersamples(:,hp);

    dist_mat_hp = ...
        matrify(@(x,y) (x-y), hypersamples_hp, hypersamples_hp);

    K_q_hp = ...
        (2*pi*width^2)^(-0.5)*exp(-0.5*(dist_mat_hp.^2/width^2));
    K_r_hp = K_q_hp;
    if grad_hyperparams
        K_r_hp = repmat(K_r_hp, num_hps+1);

        hp_block = (num_hps+1-hp)*num_data+(1:num_data);

        K_r_hp(hp_block,:) = K_r_hp(hp_block,:).*...
                            -repmat(dist_mat_hp/width^2,1,num_hps+1);
        K_r_hp(:,hp_block) = K_r_hp(:,hp_block).*...
                            repmat(dist_mat_hp/width^2,num_hps+1,1);
        K_r_hp(hp_block,hp_block) = -K_r_hp(hp_block,hp_block) + ...
                            width^-2*K_q_hp;
    end


    term = 2*prior_SD_hp^2*width^2 + width^4;
    PrecX = ...
        (prior_SD_hp^2+width^2)/term;
        %(prior_SD_hp^2+width^2-prior_SD_hp^4/(prior_SD_hp^2+width^2))^(-1);
    PrecY = ...
        -prior_SD_hp^2/term;
        %(prior_SD_hp^2-(prior_SD_hp^2+width^2)^2/(prior_SD_hp^2))^(-1);
    const = 1/(2*pi*sqrt(term));

    N_fn=@(x,y) const*...
        exp(-0.5*PrecX*((x-prior_mean_hp).^2+(y-prior_mean_hp).^2)-...
                PrecY.*(x-prior_mean_hp).*(y-prior_mean_hp));

    N_hp = matrify(@(x,y) N_fn(x,y), hypersamples_hp, hypersamples_hp);

    if grad_hyperparams
        N_hp = repmat(N_hp, 1, num_hps+1);

        N_hp(:,hp_block) = N_hp(:,hp_block).*...
                width^-2*(prior_mean_hp - ...
                    repmat(hypersamples_hp',num_data,1) + ...
                    matrify(@(x,y) (PrecX+PrecY)*prior_SD_hp^2*...
                    ((x-prior_mean_hp)+(y-prior_mean_hp)),...
                    hypersamples_hp,hypersamples_hp)...
                );



    end

    K_q = K_q.*K_q_hp;
    K_r = K_r.*K_r_hp;
    N = N.*N_hp;

end

K_q = K_q + quad_noise_sd^2*eye(num_data);
noise_vec = quad_noise_sd^2*ones(1,length(K_r));
K_r = K_r + diag(noise_vec);

allowed_error = 1e-16;
num_trials = 10;

for trial = 1:num_trials
    try
        [K_q, jitters] = improve_covariance_conditioning( ...
            K_q, ...
            [], ...
            allowed_error ...
        );
        R_q = chol(K_q);
        break
    catch
        allowed_error = allowed_error / 2;
        if trial == num_trials
            % it still didn't work!
            R_q = diag(sqrt(diag(K_q)));
        end
    end
end
for trial = 1:num_trials
    try
        [K_r, jitters] = improve_covariance_conditioning( ...
            K_r, ...
            [], ...
            allowed_error ...
        );
        R_r = chol(K_r);
        break
    catch
        allowed_error = allowed_error / 2;
        if trial == num_trials
            % it still didn't work!
            R_r = diag(sqrt(diag(K_r)));
        end
    end
end

invK_N_invK = solve_chol(R_r, solve_chol(R_q, N)')';

function x = remove_infs(x)
x = real(x);
x = max(-1e200,min(1e200,x));
