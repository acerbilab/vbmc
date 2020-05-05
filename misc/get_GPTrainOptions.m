function gptrain_options = get_GPTrainOptions(Ns_gp,hypstruct,optimState,stats,options)
%GETGPTRAINOPTIONS Get options for training GP hyperparameters.

iter = optimState.iter;
if iter > 1; rindex = stats.rindex(iter-1); else; rindex = Inf; end

gptrain_options.OutwarpFun = optimState.gpOutwarpfun;
gptrain_options.Thin = options.GPSampleThin;    % MCMC thinning
gptrain_options.InitMethod = options.GPTrainInitMethod;
gptrain_options.TolOpt = options.GPTolOpt;
gptrain_options.TolOptMCMC = options.GPTolOptMCMC;
gptrain_options.Widths = [];

% Get hyperparameter posterior covariance from previous iters
hypcov = GetHypCov(hypstruct,optimState,stats,options);    

% Set up MCMC sampler
switch lower(options.GPHypSampler)
    case {'slicesample'}
        gptrain_options.Sampler = 'slicesample';        
        if options.GPSampleWidths > 0 && ~isempty(hypcov)
            widthmult = max(options.GPSampleWidths,rindex);
            hypwidths = sqrt(diag(hypcov)');
            gptrain_options.Widths = max(hypwidths,1e-3)*widthmult;
        else
            gptrain_options.Widths = [];
        end
    case {'npv'}
        gptrain_options.Sampler = 'npv';
    case {'mala'}
        gptrain_options.Sampler = 'mala';        
        if ~isempty(hypcov)
            gptrain_options.Widths = sqrt(diag(hypcov)');
        else
            gptrain_options.Widths = [];
        end
        if isfield(optimState,'gpmala_stepsize')
            gptrain_options.Stepsize = optimState.gpmala_stepsize;
        end
    case {'slicelite'}
        gptrain_options.Sampler = 'slicelite';        
        if options.GPSampleWidths > 0 && ~isempty(hypcov)
            widthmult = max(options.GPSampleWidths,rindex);
            hypwidths = sqrt(diag(hypcov)');
            gptrain_options.Widths = max(hypwidths,1e-3)*widthmult;
        else
            gptrain_options.Widths = [];
        end
    case {'splitsample'}
        gptrain_options.Sampler = 'splitsample';        
        if options.GPSampleWidths > 0 && ~isempty(hypcov)
            widthmult = max(options.GPSampleWidths,rindex);
            hypwidths = sqrt(diag(hypcov)');
            gptrain_options.Widths = max(hypwidths,1e-3)*widthmult;
        else
            gptrain_options.Widths = [];
        end        
    case 'covsample'
        if options.GPSampleWidths > 0 && ~isempty(hypcov)
            widthmult = max(options.GPSampleWidths,rindex);
            if all(isfinite(widthmult)) && all(rindex < options.CovSampleThresh)
                nhyp = size(hypcov,1);
                gptrain_options.Widths = (hypcov + 1e-6*eye(nhyp))*widthmult^2;
                gptrain_options.Sampler = 'covsample';
                gptrain_options.Thin = gptrain_options.Thin*ceil(sqrt(nhyp));
            else
                hypwidths = sqrt(diag(hypcov)');
                gptrain_options.Widths = max(hypwidths,1e-3)*widthmult;                    
                gptrain_options.Sampler = 'slicesample';        
            end
        else
            gptrain_options.Widths = [];
            gptrain_options.Sampler = 'slicesample';        
        end
    case 'laplace'
        gptrain_options.Widths = [];
        if optimState.Neff < 30
            gptrain_options.Sampler = 'slicesample';        
            if options.GPSampleWidths > 0 && ~isempty(hypcov)
                widthmult = max(options.GPSampleWidths,rindex);
                hypwidths = sqrt(diag(hypcov)');
                gptrain_options.Widths = max(hypwidths,1e-3)*widthmult;
            end
        else
            gptrain_options.Sampler = 'laplace';
        end

    otherwise
        error('vbmc:UnknownSampler', ...
            'Unknown MCMC sampler for GP hyperparameters.');
end

% N-dependent initial training points
a = -(options.GPTrainNinit - options.GPTrainNinitFinal);
b = -3*a;
c = 3*a;
d = options.GPTrainNinit;
x = (optimState.Neff - options.FunEvalStart) / (min(options.MaxFunEvals,1e3)-options.FunEvalStart);
f = @(x) a*x.^3 + b*x.^2 + c*x + d;
Ninit = max(round(f(x)),0);

% Set other hyperparameter fitting parameters
if optimState.RecomputeVarPost
    gptrain_options.Burnin = gptrain_options.Thin*Ns_gp;
    gptrain_options.Ninit = Ninit;
    if Ns_gp > 0; gptrain_options.Nopts = 1; else; gptrain_options.Nopts = 2; end
else
    gptrain_options.Burnin = gptrain_options.Thin*3;
    if iter > 1 && stats.rindex(iter-1) < options.GPRetrainThreshold
        gptrain_options.Ninit = 0;
        if strcmpi(options.GPHypSampler,'slicelite')
            gptrain_options.Burnin = max(1,ceil(gptrain_options.Thin*log(stats.rindex(iter-1))/log(options.GPRetrainThreshold)))*Ns_gp;
            gptrain_options.Thin = 1;
        end
        if Ns_gp > 0; gptrain_options.Nopts = 0; else; gptrain_options.Nopts = 1; end            
    else
        gptrain_options.Ninit = Ninit;
        if Ns_gp > 0; gptrain_options.Nopts = 1; else; gptrain_options.Nopts = 2; end
    end
end

%gptrain_options.Burnin = 1000;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function hypcov = GetHypCov(hypstruct,optimState,stats,options)
%GETHYPCOV Get hyperparameter posterior covariance

if optimState.iter > 1
    if options.WeightedHypCov
        w_list = [];
        hyp_list = [];
        w = 1;
        for i = 1:optimState.iter-1
            if i > 1
                % diff_mult = max(1,log(stats.rindex(optimState.iter-i+1)));
                diff_mult = max(1, ...
                    log(stats.sKL(optimState.iter-i+1) ./ (options.TolsKL*options.FunEvalsPerIter)));
                w = w*(options.HypRunWeight^(options.FunEvalsPerIter*diff_mult));
            end
            if w < options.TolCovWeight; break; end     % Weight is getting too small, break

            hyp = stats.gpHypFull{optimState.iter-i};
            nhyp = size(hyp,2);
            if isempty(hyp_list) || size(hyp_list,2) == size(hyp,1)
                hyp_list = [hyp_list; hyp'];
                w_list = [w_list; w*ones(nhyp,1)/nhyp];
            end
        end
        
        w_list = w_list / sum(w_list);                  % Normalize weights
        mustar = sum(bsxfun(@times,w_list,hyp_list),1); % Weighted mean

        % Weighted covariance matrix
        nhyp = size(hyp_list,2);        
        hypcov = zeros(nhyp,nhyp);
        for j = 1:size(hyp_list,1)
            hypcov = hypcov + ...
                w_list(j)*(hyp_list(j,:)-mustar)'*(hyp_list(j,:)-mustar);            
        end
        hypcov = hypcov/(1-sum(w_list.^2));
        
    else
        hypcov = hypstruct.runcov;
    end
else
    hypcov = [];
end

end
