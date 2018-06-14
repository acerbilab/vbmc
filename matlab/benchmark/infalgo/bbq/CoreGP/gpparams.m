function covvy = gpparams(XData, YData, covvy, flag, active, samples)
% If flag=='overwrite', we ignore what is in covvy and just
% calculate and store all parameters afresh.
% active are the indices corresponding to stored cholKs etc. that are
% either new, if flag=='update', or to be removed, if flag=='downdate'.
% This function assumes that observations are corrupted by Gaussian noise
% with a single specified noise standard deviation.
% That is, XData and YData are always the correct arrays for AFTER gpparams
% is run, that is, incorporating the updated or downdate as appropriate.

if (nargin < 4)
	flag = 'overwrite';
end

flag = lower(flag);

if (nargin < 5) 
    switch flag
        case 'update'
            % assume we have added exactly one datum to the end of our YData and
            % XData
            active = length(YData); 
        case 'downdate'
        	% assume we wish to remove exactly one datum from the beginning of
            % our YData and XData
            active = 1;
        case 'overwrite'
            active = [];
    end
end

if (nargin < 6)
	samples = 1:numel(covvy.hypersamples);
end

updating = strcmp(flag, 'update');
downdating = strcmp(flag, 'downdate');
overwriting = strcmp(flag, 'overwrite');
fill_in = strcmp(flag, 'fill_in');

lowr.UT = true;
lowr.TRANSA = true;
uppr.UT = true;

[NData,NDims] = size(XData);
% if NData < NDims
%     disp('Warning: make sure each data point is a row of XData');
% end
NHyperparams = numel(covvy.hyperparams);

% if isfield(covvy,'active_hp_inds')
%     active_hp_inds=covvy.active_hp_inds;
%     inactive_hp_inds=setdiff(1:NHyperparams,active_hp_inds);
% else
%     inactive_hp_inds=[];
% end


noise_fun_test = isfield(covvy,'noisefn');
logNoiseSDPos = [];
if ~noise_fun_test 
    if (~isfield(covvy, 'logNoiseSDPos'))
        names = {covvy.hyperparams.name};
        logNoiseSDPos = cellfun(@(x) strcmpi(x, 'logNoiseSD'), names);
        covvy.logNoiseSDPos = logNoiseSDPos;
    else
        logNoiseSDPos = covvy.logNoiseSDPos;
    end
    if ~isempty(logNoiseSDPos)
        proto_noise_fun = @(hps) (@(X) exp(2 * hps(logNoiseSDPos))*eye(size(X,1)));
    else
        proto_noise_fun = @(hps) (@(X) 0);
    end
else
    proto_noise_fun = covvy.noisefn;
end

if isfield(covvy,'derivs_mean')
    derivs_mean = covvy.derivs_mean;
elseif isfield(covvy,'use_derivatives') && ~covvy.use_derivatives
    derivs_mean = false;
    covvy.derivs_mean = derivs_mean;
else
    try
        [Mu DMu] = covvy.meanfn(covvy.hypersamples(1).hyperparameters, ...
                                                        'deriv hyperparams');
        derivs_mean = true;
    catch
        derivs_mean = false;
    end
    covvy.derivs_mean = derivs_mean;
end

% is a non-constant prior mean to be specified?
mean_fun_test = isfield(covvy,'meanfn');
meanPos = [];
proto_Mu_and_DMu = 0;
if ~mean_fun_test 
    if ~isfield(covvy,'meanPos')
        names = {covvy.hyperparams.name};
        meanPos = find(cellfun(@(x) strcmpi(x, 'MeanConst'), names));
        if isempty(meanPos)
            meanPos = find(cellfun(@(x) strcmpi(x, 'Mean'), names));
        end
        if isempty(meanPos)
            meanPos = find(cellfun(@(x) strcmpi(x, 'PriorMean'), names));
        end
        covvy.meanPos = meanPos;
    else
        meanPos = covvy.meanPos;
    end
    % if meanPos is empty, this means we assume a zero mean.
    
    if ~isempty(meanPos)
        proto_Mu = @(hps) (@(X) hps(meanPos));
    else
        proto_Mu = @(hps) (@(X) 0);
    end
else

    if derivs_mean
        proto_Mu_and_DMu = covvy.meanfn;
    else
        proto_Mu = @(hps) covvy.meanfn(hps, 'deriv hyperparams');
    end
    
end




if isfield(covvy,'derivs_cov')
    derivs_cov = covvy.derivs_cov;
elseif isfield(covvy,'use_derivatives') && ~covvy.use_derivatives
    derivs_cov = false;
    covvy.derivs_cov = derivs_cov;
else
    try
        [K DK] = covvy.covfn(covvy.hypersamples(1).hyperparameters,'deriv hyperparams');
        derivs_cov=true;
    catch
        derivs_cov=false;
    end
    covvy.derivs_cov = derivs_cov;
end

proto_K_and_DK = 0;
if (derivs_cov)
    proto_K_and_DK = covvy.covfn;
else
    proto_K = @(hps) covvy.covfn(hps);
end


temp_hypersamples = covvy.hypersamples;

fill_in_test = @(sample,name) ...
    ~isfield(covvy.hypersamples(sample),name) ...
    || isempty(covvy.hypersamples(sample).(name));

names = {'logL','glogL','datahalf','datatwothirds','cholK'};
for name_ind = 1:length(names)
    name = names{name_ind};
    if fill_in_test(1,name)
        temp_hypersamples(1).(name)=[];
    end
end

for sample=samples
    
    hypersample = temp_hypersamples(sample).hyperparameters;
    

    
    if derivs_mean && mean_fun_test
        [Mu DMu] = proto_Mu_and_DMu(hypersample,'deriv hyperparams');
    else
        Mu = proto_Mu(hypersample);
    end
    YData_minus_Mu = YData - Mu(XData);
	
    if derivs_cov
        [K DK] = proto_K_and_DK(hypersample,'deriv hyperparams');
        if noise_fun_test
            [noise_fun, Dnoise_fun] = proto_noise_fun(hypersample,'deriv hyperparams');
        else
            % in this case, we'll do the derivatives below assuming the
            % noise is just a constant.
            noise_fun = proto_noise_fun(hypersample);
        end
    else
        K = proto_K(hypersample);
        noise_fun = proto_noise_fun(hypersample);
    end
	
	if (overwriting || ...
        (updating && ~isfield(temp_hypersamples(sample),'cholK')) || ...
        (fill_in && fill_in_test(sample,'cholK')))
        Kmat = improve_covariance_conditioning(...
            K(XData, XData) + noise_fun(XData),abs(YData_minus_Mu));
		cholK = chol(Kmat);
	elseif (downdating)
		cholK = downdatechol(temp_hypersamples(sample).cholK, active);
	elseif (updating && isfield(temp_hypersamples(sample), 'cholK')) 
		cholK_old = temp_hypersamples(sample).cholK;
        
        
        % set the diagonal of K_new to be the same diagonal as of our
        % previous covariance matrix -- we can cheaply reconstruct it from
        % cholK to avoid storing it.
        K_new = nan(NData);
        diag_K_new = diag_inds(K_new);
        K_new(diag_K_new(setdiff(1:end,active))) ...
            = sum(cholK_old.^2);
        K_new(active,:) = K(XData(active,:), XData);
        K_new(:,active) = K_new(active,:)';
        K_new(active,active) = K_new(active,active) + noise_fun(XData(active,:));
        K_new = improve_covariance_conditioning(K_new,abs(YData_minus_Mu));
        
        cholK = updatechol(K_new,cholK_old,active);
		
% 		two = active;
% 		xtwo = XData(two,:);
% 		one = 1:(min(two) - 1);
% 		xone = XData(one,:);
% 		three = (max(one) + 1):length(cholK_old);
% 		xthree = XData(max(two) + 1:end,:);
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

		datahalf = linsolve(cholK, YData_minus_Mu, lowr);
	elseif updating && ...
					isfield(temp_hypersamples(sample), 'datahalf')  

        datahalf_old = temp_hypersamples(sample).datahalf;
        datahalf = updatedatahalf(cholK,YData_minus_Mu,datahalf_old,cholK_old,active);
        % D = updatedatahalf(S,L,C,cholK_old,two)
        % D is inv(S')*L, and C is inv(cholK_old')*L(setdiff(1:end,two))
		
% 		old1 = covvy.hypersamples(sample).datahalf(one,:);
% 		old3 = covvy.hypersamples(sample).datahalf(three,:);
% 		
% 		new2 = linsolve(S22, YData(two) - Mean - S12' * old1, lowr);
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
        
% 		logL = covvy.hypersamples(sample).logL - 0.5 * length(active) * ...
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
	
	if (derivs_cov)
    if (overwriting || updating || ...
        (fill_in && fill_in_test(sample,'glogL')))
			DKcell = DK(XData, XData);
			% NB: K here includes the +delta(x,y)*sig^2 noise factor
            if ~noise_fun_test
                DKcell{logNoiseSDPos} = DKcell{logNoiseSDPos} + 2 * noise_fun(XData);
            else
                DKcell{logNoiseSDPos} = DKcell{logNoiseSDPos} + Dnoise_fun(XData);
            end
			
            if derivs_mean
                if (mean_fun_test)
                    DMucell = DMu(XData);
                elseif ~isempty(meanPos)
                    DMucell = mat2cell2d(zeros(NData * NHyperparams, 1), NData ...
                                                             * ones(1, NHyperparams), 1);
                    DMucell{meanPos} = ones(NData, 1);
                else
                    DMucell = mat2cell2d(zeros(NData * NHyperparams, 1), NData ...
                                                             * ones(1, NHyperparams), 1);
                end
            else
                % You're in trouble.
                DMucell = mat2cell2d(zeros(NData * NHyperparams, 1), NData ...
                                                             * ones(1, NHyperparams), 1);
            end
            
%             for hp=inactive_hp_inds
%                 DKcell{hp}=DKcell{hp}*0;
%                 DMucell{hp}=DMucell{hp}*0;
%                 % We're not interested in what's going on in these
%                 % hyperparams
%             end
			
			% These could be stored and updated should they prove
			% computationally costly to evaluate
			
			%Kinvt = (eye(NData) / cholK) / cholK';

			glogL = cellfun(@(DKmat, DMumat) -0.5 * trace(cholK\(cholK'\DKmat)) ... %Kinvt(:)' * DKmat(:) ...
											+ DMumat' * datatwothirds + 0.5* datatwothirds' ...
											* DKmat * datatwothirds, DKcell, DMucell, ...
											'UniformOutput', false);

			
			
    else
			glogL = temp_hypersamples(sample).glogL;
    end
		
    temp_hypersamples(sample).glogL = glogL;
	end
	temp_hypersamples(sample).datahalf = datahalf;
	temp_hypersamples(sample).datatwothirds = datatwothirds;
	temp_hypersamples(sample).cholK = cholK;
    temp_hypersamples(sample).hyperparameters = hypersample;
end
 
covvy.hypersamples = temp_hypersamples;

