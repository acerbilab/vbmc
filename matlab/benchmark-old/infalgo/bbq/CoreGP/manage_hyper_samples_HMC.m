function covvy = manage_hyper_samples_HMC (covvy, likelihood_fn, last_sample, hmc_options)
% Drop one hypersample, add another

%  HMC2   Hybrid Monte Carlo sampling.
% 
%        Description
%        SAMPLES = HMC2(F, X, OPTIONS, GRADF) uses a  hybrid Monte Carlo
%        algorithm to sample from the distribution P ~ EXP(-F), where F is the
%        first argument to HMC2. The Markov chain starts at the point X, and
%        the function GRADF is the gradient of the `energy' function F.
% 
%        HMC2(F, X, OPTIONS, GRADF, P1, P2, ...) allows additional arguments to
%        be passed to F() and GRADF().
% 
%        [SAMPLES, ENERGIES, DIAGN] = HMC2(F, X, OPTIONS, GRADF) also returns a
%        log of the energy values (i.e. negative log probabilities) for the
%        samples in ENERGIES and DIAGN, a structure containing diagnostic
%        information (position, momentum and acceptance threshold) for each
%        step of the chain in DIAGN.POS, DIAGN.MOM and DIAGN.ACC respectively.
%        All candidate states (including rejected ones) are stored in
%        DIAGN.POS. The DIAGN structure contains fields:
% 
%        pos
%         the position vectors of the dynamic process
%        mom
%         the momentum vectors of the dynamic process
%        acc
%         the acceptance thresholds
%        rej
%         the number of rejections
%        stp
%         the step size vectors
% 
%        S = HMC2('STATE') returns a state structure that contains the state of
%        the two random number generators RAND and RANDN and the momentum of
%        the dynamic process.  These are contained in fields  randstate,
%        randnstate and mom respectively.  The momentum state is only used for
%        a persistent momentum update.
% 
%        HMC2('STATE', S) resets the state to S.  If S is an integer, then it
%        is passed to RAND and RANDN and the momentum variable is randomised.
%        If S is a structure returned by HMC2('STATE') then it resets the
%        generator to exactly the same state.
% 
%        See HMC2_OPT for the optional parameters in the OPTIONS structure.
% 
%        See also
%        hmc2_opt, METROP

active_hp_inds = covvy.active_hp_inds;

if nargin<3 || isempty(last_sample)
last_sample = covvy.hypersamples(end).hyperparameters(active_hp_inds);
end

prior_means = cat(1,covvy.hyperparams.priorMean)';
prior_means = prior_means(active_hp_inds);
prior_SDs = cat(1,covvy.hyperparams.priorSD)';
prior_SDs = prior_SDs(active_hp_inds);
prior_C = diag(prior_SDs.^2);
prior_const = 0.5 * log(det(2 * pi * prior_C));

f = @(x) negative_log_likelihood_prior (x, covvy, likelihood_fn, 'f', prior_means, prior_C, prior_const);
g = @(x) negative_log_likelihood_prior (x, covvy, likelihood_fn, 'g', prior_means, prior_C, prior_const);

if nargin<4
    hmc_options = hmc2_opt; % default options ie. one sample
end


[new_active_samples,new_negative_log_likelihood_priors] = ...
    hmc2(f, last_sample, hmc_options, g);


num_hps = numel(covvy.hyperparams);
num_samples = numel(covvy.hypersamples);
num_new_samples = min(size(new_active_samples,1),num_samples);

new_samples = repmat(covvy.hypersamples(1).hyperparameters,num_new_samples,1);
new_samples(:,active_hp_inds) = new_active_samples;

range = 1:(num_samples - num_new_samples);
[covvy.hypersamples(range)] = covvy.hypersamples(range + num_new_samples);

new_range = (num_samples - num_new_samples + 1):num_samples;
new_samples_cell = mat2cell2d(new_samples,ones(num_new_samples,1),num_hps);

% hmc2 is actually computing the likelihoods at all those new samples, but
% they're all mixed in with the priors. We have to take the priors away
% again.
difference = (new_active_samples - repmat(prior_means,num_new_samples,1));
prior_twothirds =  difference.* repmat(prior_SDs.^-2,num_new_samples,1);
new_logLs = -new_negative_log_likelihood_priors ...
            + 0.5 * sum(prior_twothirds .* difference,2) ...
            + prior_const;
new_logLs_cell = mat2cell2d(new_logLs,ones(num_new_samples,1),1);

% Cleans off all fields of the new_samples
names = fieldnames(covvy.hypersamples);
empties = cell(num_new_samples,1);
for name_ind = 1:length(names)
    [covvy.hypersamples(new_range).(names{name_ind})]=empties{:};
end

[covvy.hypersamples(new_range).hyperparameters] = new_samples_cell{:};
[covvy.hypersamples(new_range).logL] = new_logLs_cell{:};

covvy.lastHyperSampleMoved = 1:num_samples;

function out = negative_log_likelihood_prior (x, covvy, likelihood_fn, flag, prior_means, prior_C, prior_const)

switch flag
    case 'f'
        covvy.derivs_cov = false;
    case 'g'
        covvy.derivs_cov = true;
    otherwise
        error('wrong flag')
end

active_hp_inds = covvy.active_hp_inds;

covvy.hypersamples(1).hyperparameters(active_hp_inds) = x;
%try
covvy = likelihood_fn(covvy, 1);
% catch
%     covvy.hypersamples(1).logL = -inf;
%     covvy.hypersamples(1).glogL = {zeros(max(active_hp_inds),1)};
% end

prior_twothirds = (x - prior_means) / prior_C;

if (strcmp(flag,'f'))
    out = -covvy.hypersamples(1).logL + 0.5 * prior_twothirds * (x - prior_means)' + prior_const;
elseif (strcmp(flag, 'g'))
  first_term = -cell2mat(covvy.hypersamples(1).glogL)';
  out = first_term(active_hp_inds) + prior_twothirds;
end

