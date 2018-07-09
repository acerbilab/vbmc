function gp = revise_gp(X_data, y_data, gp, flag, active, samples, grad_hp_inds)
%  gp = revise_gp(X_data, y_data, gp, flag, active, samples, grad_hp_inds)
% X_data:   (additional) evaluated inputs to condition gp on
% y_data:   (additional) evaluated outputs to condition gp on
% gp:       existing gp structure (can be empty)
% flag:     if flag=='overwrite', we ignore what is in gp and just
%           calculate and store all terms afresh (see below for further)
% active:   the indices of evaluated data that are either new, if
%           flag=='update', or to be removed, if flag=='downdate'. If
%           flag=='update', X_data and y_data can be either of full length,
%           with the new entries in positions indicated by active, or,
%           alternatively, of length exactly equal to that of active,
%           containing only new entries.
% samples:  the hyperparameter samples to revise. The default is all.
% grad_hp_inds: we compute the derivative of the log likelihood with respect to
%           these hyperparameters.



if (nargin < 6) || strcmpi(samples,'all')
	samples = 1:numel(gp.hypersamples);
end
if isempty(samples)
    return
end

if (nargin < 4) || isempty(flag)
	flag = 'overwrite';
end

flag = lower(flag);

if (nargin < 5) || isempty(active)
    switch flag
        case 'update'
            % assume we have added data to the end of our existing
            % y_data and X_data

            existing_length = length(gp.hypersamples(samples(1)).cholK);
            active = existing_length + (1:length(y_data));
        case 'downdate'
        	% assume we wish to remove exactly one datum from the beginning
        	% of
            % our y_data and X_data
            active = 1;
        case {'overwrite', 'new_hps'}
            active = [];
    end
end

if nargin > 6 && ~isempty(grad_hp_inds)
    gp.grad_hyperparams = true;
else
    if isfield(gp, 'active_hp_inds')
        grad_hp_inds = gp.active_hp_inds;
    else
        grad_hp_inds = 1:numel(gp, 'hyperparams');
    end
end

if isfield(gp, 'grad_hyperparams')
    grad_hyperparams = gp.grad_hyperparams;
else
    try
        DNoise = get_noise(gp, {'grad hyperparams',grad_hp_inds});
        DMu = get_mu(gp, {'grad hyperparams',grad_hp_inds});
        DK = gp.covfn({'grad hyperparams',grad_hp_inds});
        grad_hyperparams = true;
    catch
        grad_hyperparams = false;
    end
    gp.grad_hyperparams = grad_hyperparams;
end


Noise = get_noise(gp, 'plain');
if grad_hyperparams
    DNoise = get_noise(gp, {'grad hyperparams',grad_hp_inds});
end

Mu = get_mu(gp, 'plain');
if grad_hyperparams
    DMu = get_mu(gp, {'grad hyperparams',grad_hp_inds});
end

K = gp.covfn('plain');
if grad_hyperparams
    DK = gp.covfn({'grad hyperparams',grad_hp_inds});
end

if isfield(gp, 'abs_diffs_cov')
    % the covariance is capable of being computed given only sqd diffs
    abs_diffs_cov = gp.abs_diffs_cov;
else
    try
        K(gp.hypersamples(1).hyperparameters, abs_diffs_data);
        if grad_hyperparams
            DK(gp.hypersamples(1).hyperparameters, abs_diffs_data);
        end
        abs_diffs_cov = true;
    catch
        abs_diffs_cov = false;
    end
    gp.abs_diffs_cov = abs_diffs_cov;
end

[gp, flag] = ...
    set_gp_data(gp, X_data, y_data, flag, active);
y_data = gp.y_data;
X_data = gp.X_data;
[NData,NDims] = size(X_data);
if abs_diffs_cov
    abs_diffs_data = gp.abs_diffs_data;
end

if isempty(y_data) || isempty(X_data) || ...
        (abs_diffs_cov && isempty(abs_diffs_data))
    return
end

updating = strcmp(flag, 'update');
downdating = strcmp(flag, 'downdate');
overwriting = strcmp(flag, 'overwrite');
fill_in = strcmp(flag, 'fill_in');

lowr.UT = true;
lowr.TRANSA = true;
uppr.UT = true;


if NData < NDims
    warning('revise_gp:small_num_data', 'make sure each data point is a row of X_data');
end



temp_hypersamples = gp.hypersamples;

fill_in_test = @(sample,name) ...
    ~isfield(gp.hypersamples(sample),name) ...
    || isempty(gp.hypersamples(sample).(name));

names = {'logL','glogL','datahalf','datatwothirds','cholK'};
for name_ind = 1:length(names)
    name = names{name_ind};
    if fill_in_test(1,name)
        temp_hypersamples(1).(name)=[];
    end
end

for sample_ind = 1:length(samples)
    sample = samples(sample_ind);

    hs = temp_hypersamples(sample).hyperparameters;

    y_data_minus_Mu = y_data - Mu(hs,X_data);


	if (overwriting || ...
        (updating && ~isfield(temp_hypersamples(sample),'cholK')) || ...
        (fill_in && fill_in_test(sample,'cholK')))
        if abs_diffs_cov
            Kmat = K(hs, abs_diffs_data);
        else
            Kmat = K(hs, X_data, X_data);
        end

        allowed_error = 1e-16;
        num_trials = 10;

        for trial = 1:num_trials
            try
                [Kmat, jitters] = improve_covariance_conditioning( ...
                    Kmat + Noise(hs, X_data).^2, abs(y_data_minus_Mu), ...
                    allowed_error ...
                );
        		cholK = chol(Kmat);
                break
            catch
                allowed_error = allowed_error / 2;
                if trial == num_trials
                    % it still didn't work!

                    cholK = diag(sqrt(diag(Kmat)));
                end
            end
        end
	elseif (downdating)

        Kmat = temp_hypersamples(sample).K;
        Kmat(active,:) = [];
        Kmat(:,active) = [];

        jitters  = temp_hypersamples(sample).jitters;
        jitters(active) = [];

		cholK = downdatechol(temp_hypersamples(sample).cholK, active);
	elseif (updating && isfield(temp_hypersamples(sample), 'cholK'))

        if abs_diffs_cov
            Kvec = K(hs, abs_diffs_data(active,:,:));
        else
            Kvec = K(hs, X_data(active,:), X_data);
        end

        cholK_old = temp_hypersamples(sample).cholK;


        Kmat = nan(NData);
        diag_Kmat = diag_inds(Kmat);
        K_old = temp_hypersamples(sample).K;

        non_active_inds = setdiff(1:length(diag_Kmat),active);
        Kmat(diag_Kmat(non_active_inds)) = diag(K_old);

%         % set the diagonal of K_new to be the same diagonal as of our
%         % previous covariance matrix -- we can cheaply reconstruct it from
%         % cholK to avoid storing it.
%
%         K_new(diag_K_new(setdiff(1:end,active))) ...
%             = sum(cholK_old.^2);
        Kmat(active,:) = Kvec;
        Kmat(:,active) = Kmat(active,:)';
        Kmat(active,active) = Kmat(active,active) + ...
            Noise(hs, X_data(active,:));
        [Kmat, new_jitters] = ...
            improve_covariance_conditioning(Kmat,abs(y_data_minus_Mu), 1e-16);

        jitters = zeros(NData,1);
        jitters(non_active_inds)  = temp_hypersamples(sample).jitters;
        jitters = jitters + new_jitters;

        altered_jitter_inds = non_active_inds(new_jitters(non_active_inds)>0);

        for i = 1:length(altered_jitter_inds)
            ind = altered_jitter_inds(i);
            K_old(ind,ind) = K_old(ind,ind) + new_jitters(ind);
            cholK_old = revisechol(K_old, cholK_old, ind);
        end

        cholK = updatechol(Kmat,cholK_old,active);

% 		two = active;
% 		xtwo = X_data(two,:);
% 		one = 1:(min(two) - 1);
% 		xone = X_data(one,:);
% 		three = (max(one) + 1):length(cholK_old);
% 		xthree = X_data(max(two) + 1:end,:);
%
% 		S11 = cholK_old(one, one);
% 		S12 = linsolve(cholK_old(one, one), K(xone, xtwo), lowr);
% 		S13 = cholK_old(one, three);
% 		S22 = chol(K(xtwo, xtwo) - S12' * S12);
% 		S23 = linsolve(S22, K(xtwo, xthree) - S12' * S13, lowr);
% 		S33 = chol(cholK_old(three, three)' * cholK_old(three, three) - S23' * S23);
%
% 		cholK = [S11, S12, S13; ...
% 				  	 zeros(length(two), length(one)), S22, S23;...
% 						 zeros(length(three), length(one) + length(two)), S33];

	else
		cholK = temp_hypersamples(sample).cholK;
	end

	% we store datahalf because it is readily updated
	if (overwriting || ...
        (updating && ~isfield(temp_hypersamples(sample), 'datahalf')) || ...
        (fill_in && fill_in_test(sample,'datahalf')) || ...
			downdating)

		datahalf = linsolve(cholK, y_data_minus_Mu, lowr);
	elseif updating && ...
					isfield(temp_hypersamples(sample), 'datahalf')

        datahalf_old = temp_hypersamples(sample).datahalf;
        datahalf = updatedatahalf(cholK,y_data_minus_Mu,datahalf_old,cholK_old,active);
        % D = updatedatahalf(S,L,C,cholK_old,two)
        % D is inv(S')*L, and C is inv(cholK_old')*L(setdiff(1:end,two))

% 		old1 = gp.hypersamples(sample).datahalf(one,:);
% 		old3 = gp.hypersamples(sample).datahalf(three,:);
%
% 		new2 = linsolve(S22, y_data(two) - Mean - S12' * old1, lowr);
% 		new3 = linsolve(S33, cholK_old(three, three)' * old3 - S23' * new2, lowr);
%
% 		datahalf = [old1; new2; new3];
	else
		datahalf = temp_hypersamples(sample).datahalf;
	end

	% we store datatwothirds because it is all that is required to quickly
	% generate mean predictions for any XStar
	if (overwriting || downdating || updating || ...
        (fill_in && fill_in_test(sample,'datatwothirds')))
		datatwothirds = linsolve(cholK, datahalf, uppr);
	else
		datatwothirds = temp_hypersamples(sample).datatwothirds;
	end

	if (overwriting || ...
        (updating && ~isfield(temp_hypersamples(sample), 'logL')) || ...
        (fill_in && fill_in_test(sample,'logL')))
		logsqrtInvDetSigma = -sum(log(diag(cholK)));
		quadform = sum(datahalf.^2, 1);
		logL = -0.5 * NData * log(2 * pi) + logsqrtInvDetSigma -0.5 * quadform;
	elseif (updating && isfield(temp_hypersamples(sample), 'logL'))

        logL = updatelogL(temp_hypersamples(sample).logL,cholK,cholK_old,datahalf,datahalf_old,active);

% 		logL = gp.hypersamples(sample).logL - 0.5 * length(active) * ...
% 					 log(2 * pi) - sum(log(diag(S22))) - sum(log(diag(S33))) ...
% 					 + sum(log(diag(cholK_old(three, three)))) - 0.5 * (new2' * new2 ...
%                                                                 + new3' * new3 ...
%                                                                 - old3' * old3);
	else
		% assume we never actually want to downdate the log-likelihood - why
		% would we throw away this useful information?
		logL = temp_hypersamples(sample).logL;
	end
	temp_hypersamples(sample).logL=logL;

	if grad_hyperparams
        if (overwriting || updating || ...
            (fill_in && fill_in_test(sample,'glogL')))

                if abs_diffs_cov
                    DKcell = DK(hs, abs_diffs_data);
                else
                    DKcell = DK(hs, X_data, X_data);
                end

                % NB: K here includes the +delta(x,y)*sig^2 noise factor

                DNoisecell = DNoise(hs, X_data);

                DKcell = cellfun(@plus, DKcell, DNoisecell, ...
                                                'UniformOutput', false);
                DMucell = DMu(hs, X_data);

                glogL_fn = @(DKmat, DMumat) ...
                    -0.5 * trace(solve_chol(cholK,DKmat)) ...
                    + DMumat' * datatwothirds ...
                    + 0.5* datatwothirds' * DKmat * datatwothirds;

                glogL = cellfun(glogL_fn, ...
                    DKcell, DMucell, 'UniformOutput', false);

        else
                glogL = temp_hypersamples(sample).glogL;
        end

        temp_hypersamples(sample).glogL = glogL;
	end
	temp_hypersamples(sample).datahalf = datahalf;
	temp_hypersamples(sample).datatwothirds = datatwothirds;
	temp_hypersamples(sample).cholK = cholK;
    temp_hypersamples(sample).K = Kmat;
    temp_hypersamples(sample).jitters = jitters;
    temp_hypersamples(sample).hyperparameters = hs;
end

gp.hypersamples = temp_hypersamples;

