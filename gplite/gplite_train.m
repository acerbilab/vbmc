function [gp,hyp,output] = gplite_train(hyp0,Ns,X,y,covfun,meanfun,noisefun,s2,hprior,options)
%GPLITE_TRAIN Train lite Gaussian Process hyperparameters.
%
% Documentation to be written.
%
% Luigi Acerbi 2019


if nargin < 5; covfun = []; end
if nargin < 6; meanfun = []; end
if nargin < 7; noisefun = []; end
if nargin < 8; s2 = []; end
if nargin < 9; hprior = []; end
if nargin < 10; options = []; end

%% Assign options and defaults

% Default covariance function is squared exponential ARD
if isempty(covfun); covfun = 'seard'; end

% Default mean function is constant
if isempty(meanfun); meanfun = 'const'; end

% Default noise function (constant noise, plus provided estimated noise)
if isempty(noisefun)
    if isempty(s2); noisefun = [1 0 0]; else; noisefun = [1 1 0]; end
end

% Default options
defopts.Nopts           = 3;        % # hyperparameter optimization runs
defopts.Ninit           = 2^10;     % Initial design size for hyperparameter optimization
defopts.Thin            = 5;        % Thinning for hyperparameter sampling
defopts.Burnin          = [];       % Initial burn-in for hyperparameter sampling
defopts.DfBase          = 7;        % Default degrees of freedom for Student's t prior
defopts.Sampler         = 'slicesample';    % Default MCMC sampler for hyperparameters
defopts.Widths          = [];        % Default widths (used only for HMC sampler)
defopts.LogP            = [];        % Old log probability associated to starting points

for f = fields(defopts)'
    if ~isfield(options,f{:}) || isempty(options.(f{:}))
        options.(f{:}) = defopts.(f{:});
    end
end

% Default burn-in is proportional to the thinning factor
if isempty(options.Burnin); options.Burnin = options.Thin*Ns; end

Nopts = options.Nopts;
Ninit = options.Ninit;
Thin = options.Thin;
Burnin = options.Burnin;
DfBase = options.DfBase;
Widths = options.Widths;
LogP = options.LogP;

%% Initialize inference of GP hyperparameters (bounds, priors, etc.)

% Get covariance/noise/mean functions info
[Ncov,covinfo] = gplite_covfun('info',X,covfun,[],y);
[Nnoise,noiseinfo] = gplite_noisefun('info',X,noisefun,y,s2);
[Nmean,meaninfo] = gplite_meanfun('info',X,meanfun,y);

% Hyperparameters
if isempty(hyp0); hyp0 = zeros(Ncov+Nnoise+Nmean,1); end
Nhyp = size(hyp0,1);

LB = [];
UB = [];
if isfield(hprior,'LB'); LB = hprior.LB; end
if isfield(hprior,'UB'); UB = hprior.UB; end
if isempty(LB); LB = NaN(1,Nhyp); end
if isempty(UB); UB = NaN(1,Nhyp); end
LB = LB(:)'; UB = UB(:)';

if ~isfield(hprior,'mu'); hprior.mu = []; end
if ~isfield(hprior,'sigma'); hprior.sigma = []; end
if ~isfield(hprior,'df'); hprior.df = []; end

if numel(hprior.mu) < Nhyp; hprior.mu = [hprior.mu(:); NaN(Nhyp-numel(hprior.mu),1)]; end
if numel(hprior.sigma) < Nhyp; hprior.sigma = [hprior.sigma(:); NaN(Nhyp-numel(hprior.sigma),1)]; end
if numel(hprior.df) < Nhyp; hprior.df = [hprior.df(:); NaN(Nhyp-numel(hprior.df),1)]; end

hprior.df(isnan(hprior.df)) = DfBase;

% Set covariance/noise/mean functions hyperparameters lower bounds
LB_cov = LB(1:Ncov); idx = isnan(LB_cov); LB_cov(idx) = covinfo.LB(idx);
LB_noise = LB(Ncov+1:Ncov+Nnoise); idx = isnan(LB_noise); LB_noise(idx) = noiseinfo.LB(idx);
LB_mean = LB(Ncov+Nnoise+1:Ncov+Nnoise+Nmean); idx = isnan(LB_mean); LB_mean(idx) = meaninfo.LB(idx);

% Set covariance/noise/mean functions hyperparameters upper bounds
UB_cov = UB(1:Ncov); idx = isnan(UB_cov); UB_cov(idx) = covinfo.UB(idx);
UB_noise = UB(Ncov+1:Ncov+Nnoise); idx = isnan(UB_noise); UB_noise(idx) = noiseinfo.UB(idx);
UB_mean = UB(Ncov+Nnoise+1:Ncov+Nnoise+Nmean); idx = isnan(UB_mean); UB_mean(idx) = meaninfo.UB(idx);

% Create lower and upper bounds
LB = [LB_cov,LB_noise,LB_mean];
UB = [UB_cov,UB_noise,UB_mean];
UB = max(LB,UB);

% Plausible bounds for generation of starting points
PLB_cov = covinfo.PLB;      PUB_cov = covinfo.PUB;
PLB_noise = noiseinfo.PLB;  PUB_noise = noiseinfo.PUB;
PLB_mean = meaninfo.PLB;    PUB_mean = meaninfo.PUB;

PLB = [PLB_cov,PLB_noise,PLB_mean];
PUB = [PUB_cov,PUB_noise,PUB_mean];
PLB = min(max(PLB,LB),UB);
PUB = max(min(PUB,UB),LB);

%% Hyperparameter optimization
gptrain_options = optimoptions('fmincon','GradObj','on','Display','off');    

if Ns > 0
    gptrain_options.TolFun = 0.1;  % Limited optimization
else
    gptrain_options.TolFun = 1e-6;        
end

hyp = zeros(Nhyp,Nopts);
nll = [];

% Initialize GP
gp = gplite_post(hyp0(:,1),X,y,covfun,meanfun,noisefun,s2);

% Define objective functions for optimization
gpoptimize_fun = @(hyp_) gp_objfun(hyp_(:),gp,hprior,0,0);

% Compare old probability with new probability, check amount of change
if ~isempty(LogP) && Ns > 0
    nll = Inf(1,size(hyp0,2));
    for i = 1:size(hyp0,2); nll(i) = gpoptimize_fun(hyp0(:,i)); end    
    lnw = -nll - LogP(:)';
    w = exp(lnw - max(lnw));
    w = w/sum(w);
    ESS_frac = (1/sum(w.^2))/size(hyp0,2);
    
    ESS_thresh = 0.5;
    % Little change, keep sampling
    if ESS_frac > ESS_thresh && strcmpi(options.Sampler,'slicelite')
        Ninit = 0;
        Nopts = 0;
        if strcmpi(options.Sampler,'slicelite')
            Thin_eff = max(1,round(Thin*(1 - (ESS_frac-ESS_thresh)/(1-ESS_thresh))));
            Burnin = Thin_eff*Ns;
            Thin = 1;
        end
    end
end

% First evaluate GP log posterior on an informed space-filling design
if Ninit > 0
    % tic
    optfill.FunEvals = Ninit;
    [~,~,~,output_fill] = fminfill(gpoptimize_fun,hyp0',LB,UB,PLB,PUB,hprior,optfill);
    hyp(:,:) = output_fill.X(1:Nopts,:)';
    widths_default = std(output_fill.X,[],1);
    % toc
    
    if 0
        tic
        tprior = hprior;
        for i = 1:4
            optfill.FunEvals = ceil(Ninit/5);
            [~,~,~,output_fill2] = fminfill(gpoptimize_fun,hyp0',LB,UB,PLB,PUB,tprior,optfill);            
            hyp0 = output_fill2.X(1:Nopts,:)';
            
            % Compute vector weights
            nmu = 0.5*size(output_fill2.X,1);
            weights = log(nmu+1/2)-log(1:floor(nmu));
            weights = weights'./sum(weights);

            X_top = output_fill2.X(1:numel(weights),:);
            
            tprior.mu = sum(bsxfun(@times,weights,X_top),1);
            tprior.sigma = sqrt(sum(bsxfun(@times,weights,bsxfun(@minus,X_top,tprior.mu).^2),1));
            tprior.df = 7*ones(1,Nhyp);
        end
        hyp = output_fill2.X(1:Nopts,:)';
        widths_default = std(output_fill2.X,[],1);
        toc        
        [output_fill.fvals(1),output_fill2.fvals(1)]
    end
    
else
    if isempty(nll)
        nll = Inf(1,size(hyp0,2));
        for i = 1:size(hyp0,2); nll(i) = gpoptimize_fun(hyp0(:,i)); end
    end
    [nll,ord] = sort(nll,'ascend');
    hyp = hyp0(:,ord);
    widths_default = PUB - PLB;
end

% Check that hyperparameters are within bounds
hyp = bsxfun(@min,UB'-eps(UB'),bsxfun(@max,LB'+eps(LB'),hyp));

%tic
% Perform optimization from most promising NOPTS hyperparameter vectors
for iTrain = 1:Nopts
    nll = Inf(1,Nopts);
    try
        [hyp(:,iTrain),nll(iTrain)] = ...
            fmincon(gpoptimize_fun,hyp(:,iTrain),[],[],[],[],LB,UB,[],gptrain_options);
    catch
        % Could not optimize, keep starting point
    end
end
%toc

[~,idx] = min(nll); % Take best hyperparameter vector
hyp_start = hyp(:,idx);

% Check that starting point is inside current bounds
hyp_start = min(max(hyp_start',LB+eps(LB)),UB-eps(UB))';

logp_prethin = [];  % Log posterior of samples

%% Sample from best hyperparameter vector using slice sampling
if Ns > 0
    
    Ns_eff = Ns*Thin;   % Effective number of samples (thin after)
    
    switch lower(options.Sampler)
        case 'slicesample'
            gpsample_fun = @(hyp_) gp_objfun(hyp_(:),gp,hprior,0,1);
            sampleopts.Thin = 1;
            sampleopts.Burnin = Burnin;
            sampleopts.Display = 'off';
            sampleopts.Diagnostics = false;
            if isempty(Widths)
                Widths = widths_default; 
            else
                Widths = min(Widths(:)',widths_default);
                % [Widths; widths_default]
            end
            
            [samples,fvals,exitflag,output] = ...
                slicesamplebnd(gpsample_fun,hyp_start',Ns_eff,Widths,LB,UB,sampleopts);
            hyp_prethin = samples';
            logp_prethin = fvals;
                        
        case 'slicelite'
            gpsample_fun = @(hyp_) gp_objfun(hyp_(:),gp,hprior,0,1);
            sampleopts.Thin = 1;
            sampleopts.Burnin = Burnin;
            sampleopts.Display = 'off';
            if isempty(Widths)
                Widths = widths_default; 
            else
                Widths = min(Widths(:)',widths_default);
                % [Widths; widths_default]
            end
            
            
            
            try
            if Nopts == 0
                sampleopts.Adaptive = false;
                if size(hyp,2) < Ns_eff
                    hyp = repmat(hyp,[1,ceil(Ns_eff/size(hyp,2))]);
                    hyp = hyp(:,1:Ns_eff);
                end                
                [samples,fvals,exitflag,output] = ...
                    slicelite(gpsample_fun,hyp',Ns_eff,Widths,LB,UB,sampleopts);                
            else            
                sampleopts.Adaptive = true;
                [samples,fvals,exitflag,output] = ...
                    slicelite(gpsample_fun,hyp_start',Ns_eff,Widths,LB,UB,sampleopts);
            end
            catch
                pause
            end
            hyp_prethin = samples';
            logp_prethin = fvals;
            
        case 'covsample'
            gpsample_fun = @(hyp_) gp_objfun(hyp_(:),gp,hprior,0,1);            
            sampleopts.Thin = 1;
            sampleopts.Burnin = Burnin;
            sampleopts.Display = 'off';
            sampleopts.Diagnostics = false;
            sampleopts.VarTransform = false;
            sampleopts.InversionSample = false;
            sampleopts.FitGMM = false;
            sampleopts.TolX = 1e-80;
            sampleopts.WarmUpStages = 1;
            W = 1;
            
            samples = ...
                eissample_lite(gpsample_fun,hyp_start',Ns_eff,W,Widths,LB,UB,sampleopts);
            hyp_prethin = samples';            
            
        case 'hmc'            
            gpsample_fun = @(hyp_) gp_objfun(hyp_(:),gp,hprior,0,0);
            sampleopts.display = 0;
            sampleopts.checkgrad = 0;
            sampleopts.steps = 10;
            sampleopts.nsamples = Ns_eff;
            sampleopts.stepadj = 0.01;
            sampleopts.widths = [];
            sampleopts.nomit = Burnin;
            sampleopts.widths = Widths;
            
            [samples,fvals,diagn] = ...
                hmc2(gpsample_fun,hyp_start',sampleopts,@(hyp) gpgrad_fun(hyp,gpsample_fun));            
            hyp_prethin = samples';
            
        otherwise
            error('gplite_train:UnknownSampler', ...
                'Unknown MCMC sampler for GP hyperparameters.');
    end
    
    % Thin samples
    hyp = hyp_prethin(:,Thin:Thin:end);
    logp = logp_prethin(Thin:Thin:end);
   
else
    hyp = hyp(:,idx);
    hyp_prethin = hyp;
    logp_prethin = -nll;
    logp = -nll(idx);
end

% Recompute GP with finalized hyperparameters
gp = gp_objfun(hyp,gp,[],1);

% Additional OUTPUT struct
if nargout > 2
    output.LB = LB;
    output.UB = UB;
    output.PLB = PLB;
    output.PUB = PUB;
    output.hyp_prethin = hyp_prethin;
    output.logp = logp;
    output.logp_prethin = logp_prethin;
end


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dnlZ = gpgrad_fun(hyp,gpsample_fun)
    [~,dnlZ] = gpsample_fun(hyp);
    dnlZ = dnlZ';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [nlZ,dnlZ] = gp_objfun(hyp,gp,hprior,gpflag,swapsign)
%GPLITE_OBJFUN Objective function for hyperparameter training.

if nargin < 5 || isempty(swapsign); swapsign = 0; end

compute_grad = nargout > 1 && ~gpflag;

if gpflag
    gp = gplite_post(hyp(1:end,:),gp.X,gp.y,gp.covfun,gp.meanfun,gp.noisefun,gp.s2);
    nlZ = gp;
else

    try
        % Compute negative log marginal likelihood (without prior)
        if compute_grad
            [nlZ,dnlZ] = gplite_nlZ(hyp(1:end,:),gp,[]);
        else
            nlZ = gplite_nlZ(hyp(1:end,:),gp,[]);
        end

        % Add log prior if present, with all parameters
        if ~isempty(hprior)
            if compute_grad
                [P,dP] = gplite_hypprior(hyp,hprior);
                nlZ = nlZ - P;
                dnlZ = dnlZ - dP;
            else
                P = gplite_hypprior(hyp,hprior);
                nlZ = nlZ - P;
            end
        end

        % Swap sign of negative log marginal likelihood (e.g., for sampling)
        if swapsign
            nlZ = -nlZ;
            if compute_grad; dnlZ = -dnlZ; end
        end
        
    catch
        % Something went wrong, return NaN but try to continue
        nlZ = NaN;
        dnlZ = NaN(size(hyp));        
    end
            
end

end