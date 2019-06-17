function [gp,hyp,output] = gplite_train(hyp0,Ns,X,y,meanfun,hprior,options)
%GPLITE_TRAIN Train lite Gaussian Process hyperparameters.

% Fix functions

if nargin < 5; meanfun = []; end
if nargin < 6; hprior = []; end
if nargin < 7; options = []; end

% Default mean function is constant
if isempty(meanfun); meanfun = 'const'; end

Nopts = [];
if isfield(options,'Nopts'); Nopts = options.Nopts; end
if isempty(Nopts); Nopts = 3; end   % Number of hyperparameter optimization runs

Ninit = [];
if isfield(options,'Ninit'); Ninit = options.Ninit; end
if isempty(Ninit); Ninit = 2^10; end   % Initial design size for hyperparameter optimization

Thin = [];
if isfield(options,'Thin'); Thin = options.Thin; end
if isempty(Thin); Thin = 5; end   % Thinning for hyperparameter sampling

Burnin = [];
if isfield(options,'Burnin'); Burnin = options.Burnin; end
if isempty(Burnin); Burnin = Thin*Ns; end   % Initial design size for hyperparameter optimization

DfBase = [];
if isfield(options,'DfBase'); DfBase = options.DfBase; end
if isempty(DfBase); DfBase = 7; end   % Default degrees of freedom for Student's t prior

Sampler = [];
if isfield(options,'Sampler'); Sampler = options.Sampler; end
if isempty(Sampler); Sampler = 'slicesample'; end   % Default MCMC sampler for hyperparameters

Widths = [];
if isfield(options,'Widths'); Widths = options.Widths; end
if isempty(Widths); Widths = []; end   % Default widths (used only for HMC sampler)

LogP = [];
if isfield(options,'LogP'); LogP = options.LogP; end
if isempty(LogP); LogP = []; end   % Old log probability associated to starting points 


[N,D] = size(X);            % Number of training points and dimension
ToL = 1e-6;

X_prior = X;
y_prior = y;

Ncov = D+1;     % Number of covariance function hyperparameters

% Get mean function hyperparameter info
[Nmean,meaninfo] = gplite_meanfun([],X_prior,meanfun,y_prior);

if isempty(hyp0); hyp0 = zeros(Ncov+Nmean+1,1); end
[Nhyp,N0] = size(hyp0);      % Hyperparameters

LB = [];
UB = [];
if isfield(hprior,'LB'); LB = hprior.LB; end
if isfield(hprior,'UB'); UB = hprior.UB; end
if isempty(LB); LB = NaN(1,Nhyp); end
if isempty(UB); UB = NaN(1,Nhyp); end
LB = LB(:)'; UB = UB(:)';

if ~isfield(hprior,'mu') || isempty(hprior.mu)
    hprior.mu = NaN(Nhyp,1);
end
if ~isfield(hprior,'sigma') || isempty(hprior.sigma)
    hprior.sigma = NaN(Nhyp,1);
end
if ~isfield(hprior,'df') || isempty(hprior.df)
    hprior.df = DfBase*ones(Nhyp,1);
end
if numel(hprior.mu) < Nhyp; hprior.mu = [hprior.mu(:); NaN(Nhyp-numel(hprior.mu),1)]; end
if numel(hprior.sigma) < Nhyp; hprior.sigma = [hprior.sigma(:); NaN(Nhyp-numel(hprior.sigma),1)]; end
if numel(hprior.df) < Nhyp; hprior.df = [hprior.df(:); NaN(Nhyp-numel(hprior.df),1)]; end


% Default hyperparameter lower and upper bounds, if not specified
width = max(X_prior) - min(X_prior);
height = max(y_prior)-min(y_prior);


% Read hyperparameter bounds, if specified; otherwise set defaults
LB_ell = LB(1:D);   
idx = isnan(LB_ell);                 LB_ell(idx) = log(width(idx))+log(ToL);
LB_sf = LB(D+1);        if isnan(LB_sf); LB_sf = log(height)+log(ToL); end
LB_sn = LB(Ncov+1);     if isnan(LB_sn); LB_sn = log(ToL); end

% Set mean function hyperparameters lower bounds
LB_mean = LB(Ncov+2:D+2+Nmean);
idx = isnan(LB_mean);
LB_mean(idx) = meaninfo.LB(idx);

UB_ell = UB(1:D);   
idx = isnan(UB_ell);    UB_ell(idx) = log(width(idx)*10);
UB_sf = UB(D+1);        if isnan(UB_sf); UB_sf = log(height*10); end
UB_sn = UB(Ncov+1);     if isnan(UB_sn); UB_sn = log(height); end

% Set mean function hyperparameters upper bounds
UB_mean = UB(Ncov+2:D+2+Nmean);
idx = isnan(UB_mean);
UB_mean(idx) = meaninfo.UB(idx);

% Create lower and upper bounds
LB = [LB_ell,LB_sf,LB_sn,LB_mean];
UB = [UB_ell,UB_sf,UB_sn,UB_mean];
UB = max(LB,UB);

% Plausible bounds for generation of starting points
PLB_ell = log(width)+0.5*log(ToL);
PUB_ell = log(width);

PLB_sf = log(height)+0.5*log(ToL);
PUB_sf = log(height);

PLB_sn = 0.5*log(ToL);
PUB_sn = log(std(y_prior));

PLB_mean = meaninfo.PLB;
PUB_mean = meaninfo.PUB;

PLB = [PLB_ell,PLB_sf,PLB_sn,PLB_mean];
PUB = [PUB_ell,PUB_sf,PUB_sn,PUB_mean];

PLB = min(max(PLB,LB),UB);
PUB = max(min(PUB,UB),LB);

gptrain_options = optimoptions('fmincon','GradObj','on','Display','off');    

%% Hyperparameter optimization
if Ns > 0
    gptrain_options.TolFun = 0.1;  % Limited optimization
else
    gptrain_options.TolFun = 1e-6;        
end

hyp = zeros(Nhyp,Nopts);
nll = [];

% Initialize GP
gp = gplite_post(hyp0(:,1),X,y,meanfun);

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
    if ESS_frac > ESS_thresh && strcmpi(Sampler,'slicelite')
        Ninit = 0;
        Nopts = 0;
        if strcmpi(Sampler,'slicelite')
            Thin_eff = max(1,round(Thin*(1 - (ESS_frac-ESS_thresh)/(1-ESS_thresh))));
            Burnin = Thin_eff*Ns;
            Thin = 1;
        end
    end
end

% First evaluate GP log posterior on an informed space-filling design
if Ninit > 0
    optfill.FunEvals = Ninit;
    [~,~,~,output_fill] = fminfill(gpoptimize_fun,hyp0',LB,UB,PLB,PUB,hprior,optfill);
    hyp(:,:) = output_fill.X(1:Nopts,:)';
    widths_default = std(output_fill.X,[],1);
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
    
    switch lower(Sampler)
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


% Check GP posteriors
% for s = 1:numel(gp.post)
%     if ~all(isfinite(gp.post(s).L(:)))
%         pause
%     end
% end


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
    gp = gplite_post(hyp(1:end,:),gp.X,gp.y,gp.meanfun);
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
    
%     if compute_grad
%         dnlZ
%     end
        
end

end