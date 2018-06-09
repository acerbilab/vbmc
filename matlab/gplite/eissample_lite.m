function [samples,logP,exitflag,output,fvals] = eissample_lite(logPfuns,x0,N,K,widths,LB,UB,options,varargin)
%EISSAMPLE Ensemble inversion slice sampling and other sampling methods (INTERNAL VERSION).
%
%   SAMPLES = EISSAMPLE(LOGF,X0,N) draws N random samples from a target 
%   distribution with the log probability density function (pdf) LOGPFUNS 
%   using a mixture of ensemble (slice) sampling methods. 
%   X0 is a row vector or a matrix in which each row contains an initial 
%   starting point for the the random sample sequences. Each initial 
%   starting point must be within the domain of the target distribution.
%   By default, EISSAMPLE uses an ensemble of K = 2*(D+1) walkers, where D 
%   is the number of dimensions of the target distribution (columns of X0).
%   If X0 contains more rows than walkers, the algorithm takes only the 
%   first K rows. If X0 contains less rows than walkers, additional initial 
%   vectors are created by jittering the initial points by a small amount.
%   N is the number of samples to be returned.
%   LOGPFUNS is either a function handle created using @, or a cell array
%   of function handles. Functions in LOGPFUNS take one row array as input 
%   that has the same type and number of columns as X0 and returns the 
%   target log density function (the normalization constant of the pdf need 
%   not be known). The log density returned by each function can either be 
%   a scalar or a row vector (one element per data point). If multiple 
%   function handles are provided, they are evaluated sequentially and the 
%   log probability is summed over them. 
%   If a function returns -Inf, the point is considered invalid, and next 
%   functions in the sequence are not evaluated. In Bayesian inference, 
%   typically the first function provided is the prior (which may also 
%   encode nonlinear bounds), and the second function is the likelihood.
%   SAMPLES is a matrix each row of which corresponds to a sampled point in 
%   the sequence.
%
%   SAMPLES = EISSAMPLE(LOGF,X0,N,K) uses K walkers. The default number of
%   walkers is K = 2*(D+1). It is not recommended to use less than K = 2*D
%   walkers. More walkers may better explore the posterior landscape, but
%   convergence of the chains may take longer.
%
%   SAMPLES = EISSAMPLE(LOGF,X0,N,K,WIDTHS) uses WIDTHS as a scalar or 
%   vector of typical widths. If it is a scalar, all dimensions are assumed 
%   to have the same typical widths. If it is a vector, each element of the 
%   vector is the typical width of the marginal target distribution in that 
%   dimension. The default value of W(i) is (UB(i)-LB(i))/2 if the i-th 
%   bounds are finite, or 10 otherwise. By default EISSAMPLE uses an 
%   adaptive widths method during warmup, so the choice of typical widths 
%   is not crucial.
%
%   SAMPLES = EISSAMPLE(LOGF,X0,N,K,WIDTHS,LB,UB) defines a set of lower 
%   and upper bounds on the domain of the target density function, which is
%   assumed to be zero outside the range LB <= X <= UB. Use empty matrices 
%   for LB and UB if no bounds exist. Set LB(i) = -Inf if X(i) is unbounded 
%   below; set UB(i) = Inf if X(i) is unbounded above. If LB(i) == UB(i),
%   the variable is assumed to be fixed on that dimension.
%
%   SAMPLES = EISSAMPLE(LOGF,X0,N,WIDTHS,LB,UB,OPTIONS) samples with 
%   the default sampling parameters replaced by values in the structure 
%   OPTIONS. EISSAMPLE uses these options:
%
%     OPTIONS.Thin generates random samples with Thin-1 out of Thin values 
%     omitted in the generated sequence (after warmup). Thin is a positive
%     integer. The default value of Thin is 1.
%
%     OPTIONS.Burnin omits the first Burnin points before starting recording 
%     points for the generated sequence. Burnin is a non-negative integer. 
%     The default value of Burnin is round(2*N) (that is, double the number
%     of recorded samples).
%
%     OPTIONS.Display defines the level of display. Accepted values for
%     Display are 'iter', 'notify', 'final', and 'off' for no display. The 
%     default value of Display is 'notify'. 
%    
%     OPTIONS.Diagnostics specifies whether convergence diagnostics are
%     performed at the end of the run. Diagnostics is a boolean value. Set 
%     OPTIONS.Diagnostics to true to run the diagnostics, false to skip it. 
%     The default for OPTIONS.Diagnostics is true. The diagnostics tests 
%     use the PSRF function by Simo Särkkä and Aki Vehtari, which implements 
%     diagnostics from Gelman et al. (2013).
%
%     OPTIONS.Noise is a flag that specifies whether the computation of the
%     target pdf is stochastic (noisy). The MCMC procedure is still 
%     asymptotically correct if LOGPFUNS returns an unbiased estimator of 
%     the pdf. The default value of Noise is false.
%
%   SAMPLES = EISSAMPLE(...,VARARGIN) passes additional arguments
%   VARARGIN to LOGF.
%
%   [SAMPLES,LOGP] = EISSAMPLE(...) returns the sequence of values 
%   LOGP of the target log pdf at the sampled points (sum over functions,
%   if multiple function handles are provided in LOGPFUNS).
%
%   [SAMPLES,LOGP,EXITFLAG] = EISSAMPLE(...) returns an EXITFLAG that
%   describes the exit condition of EISSAMPLE. Possible values of 
%   EXITFLAG and the corresponding exit conditions are
%
%    1  Target number of recorded samples reached, with no explicit
%       violation of convergence (this does not ensure convergence).
%    0  Target number of recorded samples reached, convergence status is
%       unknown (no diagnostics have been run).
%    -1 No explicit violation of convergence detected, but the number of
%       effective (independent) samples in the sampled sequence is much 
%       lower than the number of requested samples N for at least one 
%       dimension.
%    -3 Detected probable lack of convergence of the sampling procedure.
%    -4 Detected lack of convergence of the sampling procedure.
%    -5 One of more slice sampling iterations failed by shrinking back to
%       the point. This flag is returned only if OPTIONS.Noise is 0; in 
%       which case it usually means that there is some problem with the 
%       target distribution. For noisy functions it is a normal occurrence 
%       and it is not reported.
%
%   [SAMPLES,LOGP,EXITFLAG,OUTPUT] = EISSAMPLE(...) returns a structure
%   OUTPUT with the number of evaluations of LOGF in OUTPUT.funccount, the 
%   value WIDTHS used during sampling in OUTPUT.widths (they can differ from 
%   the initial WIDTHS due to width adaptation during warmup). OUTPUT also
%   contains results of the convergence diagnostics, if available, such as 
%   the Potential Scale Reduction Factor in OUTPUT.R, the number of 
%   effective samples in OUTPUT.Neff and the estimated autocorrelation time
%   in OUTPUT.tau.
%
%   [SAMPLES,LOGP,EXITFLAG,OUTPUT,FVALS] = EISSAMPLE(...) returns the
%   values of the functions LOGPFUNS for each sample. If LOGPFUNS is a cell
%   array, FVALS is a cell array, where the m-th cell contains a matrix
%   of values returned by the m-th function, each row corresponding to
%   a sample and columns corresponding to data points (if LOGPFUNS returns
%   the log pdf per data point).
%   Knowing the log pdf of the sampled points per each data point can be 
%   useful to compute estimates of predictive error such as the widely 
%   applicable information criterion (WAIC); see Watanabe (2010).
%
%   References: 
%   - R. Neal (2003), Slice Sampling, Annals of Statistics, 31(3), p705-67.
%   - D. J. MacKay (2003), Information theory, inference and learning 
%     algorithms, Cambridge university press, p374-7.
%   - S. Watanabe (2010), Asymptotic equivalence of Bayes cross validation
%     and widely applicable information criterion in singular learning 
%     theory, The Journal of Machine Learning Research, 11, p3571-94.
%   - A. Gelman, et al (2013), Bayesian data analysis. Vol. 2. Boca Raton, 
%     FL, USA: Chapman & Hall/CRC.

% Author:   Luigi Acerbi
% Email:    luigi.acerbi@{nyu.edu,gmail.com}
% Date:     Jun 2016

% TODO LIST:
% - Fix 1D variational fit
% - Implement https://gist.github.com/jtravs/5327056
%  1. G. E. Alefeld, F. A. Potra, and Y. Shi, "Algorithm 748: enclosing zeros of
%  # continuous functions," ACM Trans. Math. Softw. 21, 327–344 (1995).
% - Fix WSIZE choice and WIDTHS
% - Speed up warmup
% - GMM1DJINV: improve initial Gaussian approximation for ADS
% - For transSliceSampleCW and transSliceSampleRD do not choose WIDTH based
%   on single random pair but choose max after resampling a few?
% - Fix Gaussification when LB and/or UB are not specified

% Use persistent variable to avoid overhead for repeated usage
% persistent STARTUP;
% 
% if isempty(STARTUP)
%     % Add subdirectories to path
%     basedir = fileparts(mfilename('fullpath'));
%     addpath([basedir filesep 'diag'],[basedir filesep 'trans'],[basedir filesep 'utils'],[basedir filesep 'vbgmm']);
%     STARTUP = 1;
% end

%% Default parameters and options

% By default unbounded sampling
if nargin < 4; K = []; end
if nargin < 5; widths = []; end
if nargin < 6 || isempty(LB); LB = -Inf; end
if nargin < 7 || isempty(UB); UB = Inf; end
if nargin < 8; options = []; end

% Default options
defopts.Thin        = 1;            % Thinning
defopts.Burnin      = 2*N;          % Burn-in
defopts.Display     = 'notify';     % Display
defopts.Diagnostics = true;         % Convergence diagnostics at the end
defopts.Temperature = 1;
defopts.TolX        = 1e-10;
defopts.Noise       = false;        % Noisy estimate of target function
defopts.Metropolis  = true;         % Perform Metropolis step
defopts.LoadFile    = '';           % Load interrupted sampling from file
defopts.SaveFile    = '';           % Save sampling to file
defopts.SaveTime    = 1e4;          % Save every this number of seconds
defopts.VarTransform = true;        % Transform constrained variables


%--------------------------------------------------------------------------
% Internal defaults (average user should not modify these)

% Minimum weight assigned to the global component for inversion slice sampling
defopts.MinWeightGlobalComponent = 0.1;
% Standard deviations of adaptive widths
defopts.SigmaFactor = 5;
% Warmup is divided in this number of stages
defopts.WarmUpStages = 10;
% Fit surrogate Gaussian mixture model to the data during warmup
defopts.FitGMM = true;
% Use inversion sampling
defopts.InversionSample = true;
% Perform stepping-out when the current window does not bracket the pdf
defopts.StepOut     = false;
% Adaptive widths
defopts.AdaptiveWidths  = true;
% Warmup stages at which variable transform is computed (empty here)
defopts.VarTransformStages = [];
% Number of starting points for transform optimization
defopts.VarTransformRestarts = 1000;
% Method for variable transform (1: Kumaraswamy-logistic; 2: Kumaraswamy-logistic-power; 3: nonparametric CDF; 4 GMM)
defopts.VarTransformMethod = 4;

% Transition operators
defopts.TransitionOperators = {@transSliceSampleRD};

% defopts.TransitionOperators = ...
%     {@transConcat, {@transMetropolisNoiseUpdate, ...
%     @transMetropolisVBGMM, ...
%     {@transRoulette, [0.25 0.25 0.25 0.25], ...
%         {@transSliceSampleCW, ...
%         @transSliceSampleRD, ...
%         @transSliceSampleADS, ...
%         {@transSliceSampleMosaic, 0.05}}}}};

for f = fields(defopts)'
    if ~isfield(options,f{:}) || isempty(options.(f{:}))
        options.(f{:}) = defopts.(f{:});
    end
end

switch options.Display
    case {'notify','notify-detailed'}
        trace = 2;
    case {'none', 'off'}
        trace = 0;
    case {'iter','iter-detailed'}
        trace = 3;
    case {'final','final-detailed'}
        trace = 1;
    otherwise
        trace = 1;
end

% Define essential variables
[m0,nvars] = size(x0);
if isempty(K); K = 2*(nvars+1); end     % Default number of walkers
% if isempty(K); K = 10 + nvars*5; end

% Options for constrained variables transform
trinfo = [];
if options.VarTransform && isempty(options.VarTransformStages)
    options.VarTransformStages = ...
        unique([0,ceil(options.WarmUpStages*0.75),options.WarmUpStages]);
end

if ~isempty(options.LoadFile) && exist(options.LoadFile,'file')
    %% Load interrupted execution from file
    
    optionsnew = options;           % Store current options    
    if options.VarTransform; logPfuns_temp = logPfuns; end    
    load(options.LoadFile,'-mat');  % Load run from recovery file
    i0 = ii;                        % Start from recovered iteration
    if trace > 1; fprintf('Loading sampling from file ''%s''.\n', options.LoadFile); end

    % Copy some new OPTIONS to the old options structure
    copyfields = {'LoadFile','SaveFile','SaveTime','Diagnostics','Display'};    
    for f = 1:numel(copyfields)    
        options.(copyfields{f}) = optionsnew.(copyfields{f});
    end
    
    % Check that these fields exist and are nonempty (for retrocompatibility)
    checkfields = {'VarTransform','VarTransformRestarts','VarTransformMethod','VarTransformStages'};
    for f = 1:numel(checkfields)
        if ~isfield(options,checkfields{f}) || isempty(options.(checkfields{f}))
            options.(checkfields{f}) = optionsnew.(checkfields{f});
        end
    end
    
    %----------------------------------------------------------------------
    % TEMPORARY PATCH FOR RETROCOMPATIBILITY
    if ~isempty(trinfo) && options.VarTransformMethod == 4 ...
            && isfield(trinfo,'gmm') && ~isempty(trinfo.gmm)
        for i = 1:numel(trinfo.gmm)
            trinfo.gmm{i}.iMax = gmm1cdf(-1,trinfo.gmm{i}.w,-trinfo.gmm{i}.Mu,trinfo.gmm{i}.Sigma);
        end
    end
    %----------------------------------------------------------------------
    
    % Reload function handles
    if options.VarTransform
        logPfuns_orig = logPfuns_temp;
        [logPfuns,nocell_tr] = trans_logPfuns(logPfuns_orig,trinfo);
    end
        
    clear copyfields f optionsnew logPfuns_temp;
    
else
    %% Startup and initial checks
    
    % Provided WIDTHS is a covariance matrix
    if ~isempty(widths) && ~isvector(widths)
        covmat = widths;
        widths = sqrt(diag(covmat))';
    else
        covmat = [];
    end    
    
    % Transform constrained variables
    if options.VarTransform
        logPfuns_orig = logPfuns;
        [trinfo,x0,widths,LB,UB,logPfuns,nocell_tr] = ...
            trans_init(logPfuns,x0,K,widths,LB,UB,trace,options);
    end
    
    % Initialize algorithm variables
    [sampleState,logPfuns,nocell] = ...
        sampleinit(logPfuns,nvars,N,LB,UB,widths,covmat,options);

    if m0 > K
        [sampleState.gmm,sampleState.vbmodel] = fitgmm(x0,1,options);
        if m0 <= N              % Take at most N points
            warmup = x0;
        else
            ord = randperm(m0);
            warmup = x0(ord(1:N),:);
        end
    else
        warmup = [];
    end
        
    % Initial population
    x0 = sampleinitpop(logPfuns,x0,K,sampleState,options,trace,varargin);
        
    % Evaluate starting points
    ensemble.x = x0;
    ensemble.logP = zeros(K,1);
    [ensemble.logP(1),ftemp,fc] = feval(@logpdfbnd,ensemble.x(1,:),logPfuns,sampleState.LB,sampleState.UB,options,varargin{:});
    sampleState.funccount = sampleState.funccount + fc;
    for j = 1:numel(logPfuns)
        if isempty(ftemp{j}); error('Starting point should evaluate to a finite value (no NaN or Inf).'); end
        ensemble.fval{j} = zeros(K,numel(ftemp{j}));
    end
    ensemble.fval = assignfval(ensemble.fval,ftemp,1);
    for k = 2:K
        [ensemble.logP(k),ftemp,fc] = feval(@logpdfbnd,ensemble.x(k,:),logPfuns,sampleState.LB,sampleState.UB,options,varargin{:});
        sampleState.funccount = sampleState.funccount + fc;
        ensemble.fval = assignfval(ensemble.fval,ftemp,k);
    end    
    
    % Make space for stored variables
    samples = zeros(N, nvars);
    if nargout > 1; logP = zeros(N, 1); end
    if nargout > 4
        for j = 1:numel(ensemble.fval)
            fvals{j} = zeros(N, size(ensemble.fval{j},2));            
        end
    elseif nargout > 1 && ~isempty(trinfo)  % Record log transorm
        fvals{1} = zeros(N,1);            
    end

    % Check arguments and options
    sampleargcheck(logPfuns,ensemble.x,ensemble.logP,N,sampleState,warmup,options);
    if options.Burnin == 0 && isempty(sampleState.widths) && options.AdaptiveWidths
        warning('WIDTHS not specified and adaptation is ON (OPTIONS.AdaptiveWidths == 1), but OPTIONS.Burnin is set to 0. SLICESAMPLEBND will attempt to use default values for WIDTHS.');
    end

    i0 = 1;
    idx = 1;
    xx_sum = zeros(1,nvars);
    xx_sqsum = zeros(1,nvars);
    
    % Warmup steps
    warmiters = round(linspace(sampleState.burn/options.WarmUpStages, ...
        sampleState.burn,options.WarmUpStages));
end

if trace > 1
    fprintf(' Iteration     f-count       log p(x)                   Action\n');
    displayFormat = ' %7.0f     %8.0f    %12.6g    %26s\n';
end

lastsave = tic;    % Keep track of time

% Sampling function
fun = @(x_) feval(@logpdfbnd,x_,logPfuns,sampleState.LB,sampleState.UB,options,varargin{:});

% Main loop
for ii = i0:sampleState.totN
        
    % Save current progress to file
    if ~isempty(options.SaveFile) && (toc(lastsave) > options.SaveTime ||  ii == sampleState.totN)
        if trace > 1; fprintf('Saving temp file ''%s''...\n', options.SaveFile); end
        filename = options.SaveFile;
        
        if ~exist(filename,'file')
            save(filename);
        else

            % File already exists, move it to temporary copy
            [pathstr,name,ext] = fileparts(filename);
            tempfile = fullfile(pathstr,[name '.old_']);
            movefile(filename,tempfile);
            for iAttempt = 1:3
                temp = [];
                try
                    save(filename);
                    temp = load(filename,'-mat'); % See if it worked
                    if ~isempty(temp); break; end
                catch
                    % Did not work
                end
            end
            if isempty(temp)
                movefile(tempfile,filename);
                error(['Cannot save file ''' filename '''.']);
            end

            % Remove temporary old file
            if exist(tempfile,'file'); delete(tempfile); end
        end
        
        lastsave = tic;
    end
    
    % Current walker index (from 1 to K)
    idx = mod(idx, K) + 1;
    xx = ensemble.x(idx,:);
    sampleState.others = ensemble.x;
    sampleState.others(idx,:) = [];
    
    if trace > 1 && ii == sampleState.burn+1
        action = 'start recording';
        fprintf(displayFormat,ii-sampleState.burn,sampleState.funccount(end),ensemble.logP(idx),action);
    end
    
    % Independent Metropolis step
    % if ii < sampleState.burn/2; mhfrac = 0.5; elseif ii < sampleState.burn; mhfrac = 1; else mhfrac = 1; end

    %% Apply transition operators
    if trace > 1
        printfun = @(action) fprintf(displayFormat,ii-sampleState.burn,sampleState.funccount(end),ensemble.logP(idx),action);
    else
        printfun = @(action) 0;
    end
    
    [xx,ensemble.logP(idx),ftemp,exitflag,sampleState] = ...
        feval(options.TransitionOperators{:}, ...
        fun,xx,ensemble.logP(idx), ...
        cellgetrow(ensemble.fval,idx),sampleState,options,printfun);
    
    if exitflag == -1
        error('Transition operator failed.');
    end

    ensemble.fval = assignfval(ensemble.fval,ftemp,idx);
    
    % Something may be wrong with logP, try to update
    if exitflag == 1
        templogP = sum(cellfun(@sum,ftemp));
        if abs(ensemble.logP(idx)-templogP) > eps
            ensemble.logP(idx) = templogP;
            printfun = @(action) fprintf(displayFormat,ii-sampleState.burn,sampleState.funccount(end),ensemble.logP(idx),action);
            printfun('fixed potential numerical error');
        end
    end
    
    %% Record samples and miscellaneous bookkeeping    
    ensemble.x(idx,:) = xx;
    
    % Record samples?
    record = ii > sampleState.burn && mod(ii - sampleState.burn - 1, sampleState.thin) == 0;
    if record
        ismpl = 1 + (ii - sampleState.burn - 1)/sampleState.thin;
        samples(ismpl,:) = xx;
        if nargout > 1; logP(ismpl) = ensemble.logP(idx); end
        if nargout > 4
            for j = 1:numel(ensemble.fval)
                fvals{j}(ismpl,:) = ensemble.fval{j}(idx,:);
            end
        elseif nargout > 1 && ~isempty(trinfo)  % Record log transorm
            fvals{1}(ismpl,:) = ensemble.fval{1}(idx,:);
        end
    end
    
    if ii <= sampleState.burn
        warmup(end+1,:) = xx;        
        warmiter = find(ii == warmiters,1);
        
        if ~isempty(warmiter)
            action = ['end warmup stage #' num2str(warmiter)];
            if trace > 1
                fprintf(displayFormat,ii-sampleState.burn,sampleState.funccount(end),ensemble.logP(idx),action);
            end
            
            if options.VarTransform && any(warmiter == options.VarTransformStages)
                warmup = pdftrans(warmup,'inv',trinfo);
                ensemble.x = pdftrans(ensemble.x,'inv',trinfo);
                ensemble.logP = ensemble.logP - ensemble.fval{1};
                [trinfo,warmup] = trans_gauss(warmup,trinfo,options);
                trinfo
                ensemble.x = pdftrans(ensemble.x,'dir',trinfo);
                ensemble.fval{1} = pdftrans(ensemble.x,'logpdf',trinfo);
                ensemble.logP = ensemble.logP + ensemble.fval{1};                
                [logPfuns,nocell_tr] = trans_logPfuns(logPfuns_orig,trinfo);                
                fun = @(x_) feval(@logpdfbnd,x_,logPfuns,sampleState.LB,sampleState.UB,options,varargin{:});
            end
            
            if options.FitGMM
                [sampleState.gmm,sampleState.vbmodel] = fitgmm(warmup,warmiter,options);
            end
            
            % Update WIDTHS if using adaptive method
            if options.AdaptiveWidths
                newwidths = options.SigmaFactor*std(warmup);
                newwidths = min(newwidths, sampleState.UB - sampleState.LB);
                if ~isreal(newwidths); newwidths = sampleState.widths; end
                if isempty(sampleState.basewidths)
                    sampleState.widths = newwidths;
                else
                    % Max between new widths and geometric mean with user-supplied 
                    % widths (i.e. bias towards keeping larger widths)
                    sampleState.widths = max(newwidths, sqrt(newwidths.*sampleState.basewidths));
                end
                % sampleState.widths
            end
            
            if warmiter > 0; warmup(1:floor(sampleState.burn/numel(warmiters)/2),:) = []; end
            
        end
    end
    
    if trace > 2
        if ii <= sampleState.burn; action = 'burn';
        elseif ~record; action = 'thin';
        else action = 'record';
        end
        fprintf(displayFormat,ii-sampleState.burn,sampleState.funccount,ensemble.logP(idx),action);
    end    
    
end

if trace > 0
    if sampleState.thin > 1
        thinmsg = ['\n   and keeping 1 sample every ' num2str(sampleState.thin) ', '];
    else
        thinmsg = '\n   ';
    end    
    fprintf(['\nSampling terminated:\n * %d samples obtained after a burn-in period of %d samples' thinmsg 'for a total of %d full function evaluations.'], ...
        N, sampleState.burn, sampleState.funccount(end));
end

% Diagnostics
if options.Diagnostics && (nargout > 2 || trace > 0)
    [exitflag,R,Neff,tau] = samplediagnose(K,samples,sampleState,trace,options);
    diagstr = [];
    if exitflag == -2 || exitflag == -3 || exitflag == -4
        diagstr = [diagstr '\n * Try sampling for longer, by increasing N or OPTIONS.Thin.'];
    elseif exitflag == -1
        diagstr = [diagstr '\n * Try increasing OPTIONS.Thin to obtain more uncorrelated samples.'];
    elseif exitflag == 1
        diagstr = '\n * No violations of convergence have been detected (this does NOT guarantee convergence).';
    end
    if trace > 0 && ~isempty(diagstr)
        fprintf(diagstr);
    end
end

% Convert variables back to original (constrained) space if needed
if ~isempty(trinfo)
    samples = pdftrans(samples,'inverse',trinfo);
    if nargout > 1; logP = logP - fvals{1}; end
    if nargout > 3; sampleState.funccount(1) = []; end
    if nargout > 4; fvals(1) = []; end
end

if trace > 0
    fprintf('\n\n');
end

if nargout > 3
    output.widths = sampleState.widths;
    output.funccount = sampleState.funccount;
    if options.Diagnostics
        output.R = R;
        output.Neff = Neff;
        output.tau = tau;
    end
    % output.warmup = warmup;
    if isfield(sampleState,'vbmodel'); output.vbmodel = sampleState.vbmodel; end
    if ~isempty(trinfo); output.trinfo = trinfo; end
end

% Remove cell if original LOGFUN input is not cell
if nargout > 4 && (nocell || nocell_tr); fvals = fvals{1}; end

end

%--------------------------------------------------------------------------
function [gmm,vbmodel] = fitgmm(X,iter,options)
%FITGMM Fit variational Gaussian mixture model to X
    [N,nvars] = size(X);    
    if N < nvars; gmm = []; vbmodel = []; return; end
    
    prior.alpha = 0.1;
    prior.nu = nvars+1;
    
    prior.M = prior.nu*cov(X);
    optvb.Display = 'off';
    optvb.ClusterInit = 'kmeans';
    if iter < 5; optvb.Nstarts = 2;
    elseif iter < 10; optvb.Nstarts = 3;
    else optvb.Nstarts = 5;
    end
        
    vbmodel = vbgmmfit(X',10+5*nvars,prior,optvb);
    gmm = vbgmmgmm(vbmodel,'mean');
    
    % Add global component
    gmm.w(end+1) = 1/iter;
    gmm.w = gmm.w/sum(gmm.w);
    gmm.Mu(end+1,:) = mean(X);
    gmm.Sigma(:,:,end+1) = options.SigmaFactor*chol(cov(X));
end

%--------------------------------------------------------------------------
function [trinfo,x0,widths,LB,UB,logPfuns,nocell_tr] = trans_init(logPfuns,x0,K,widths,LB,UB,trace,options)
%TRANS_INIT Initialize nonlinear transformation of variables

    [m0,nvars] = size(x0);
    trinfo = pdftrans(nvars,LB,UB);
                
    if any(trinfo.type > 0)
        if trace > 1; fprintf('Applying nonlinear transformation to constrained variables.\n'); end
        
        if any(options.VarTransformStages == 0) && m0 > K
            [trinfo,x0] = trans_gauss(x0,trinfo,options);
        else
            x0 = pdftrans(x0,'d',trinfo);
        end

        % Move starting points that are too close to bounds
        newbound = abs(log(sqrt(eps)));
        x0(bsxfun(@and, trinfo.type == 1 | trinfo.type == 3, x0 < -newbound)) = log(sqrt(eps));
        x0(bsxfun(@and, trinfo.type == 2 | trinfo.type == 3, x0 > newbound)) = -log(sqrt(eps));
        
        % Convert WIDTHS to unconstrained space
        if ~isempty(widths)
            if isscalar(widths); widths = widths*ones(1,nvars); end
            y0mean = mean(x0,1);
            x0w = pdftrans(y0mean,'i',trinfo);
            for i = 1:nvars
                if ~isfinite(widths(i)); continue; end
                switch trinfo.type(i)
                    case 1
                        x0w(i) = x0w(i) + widths(i);
                        y0w = pdftrans(x0w,'d',trinfo);
                        widths(i) = max(y0w(i) - y0mean(i),1);
                    case 2
                        x0w(i) = x0w(i) - widths(i);
                        y0w = pdftrans(x0w,'d',trinfo);
                        widths(i) = max(y0w(i) - y0mean(i),1);                        
                    case 3
                        x0w2 = x0w;
                        x0w(i) = min(x0w(i) + widths(i)/2, UB(i) - eps(LB(i)));
                        x0w2(i) = max(x0w2(i) - widths(i)/2, LB(i) + eps(UB(i)));
                        y0w = pdftrans(x0w,'d',trinfo);
                        y0w2 = pdftrans(x0w2,'d',trinfo);
                        widths(i) = max(y0w(i) - y0w2(i),1);
                end
            end
        end
        
        % Switch to unconstrained space
        LB(trinfo.type > 0) = -Inf;
        UB(trinfo.type > 0) = Inf;
        
        % Add log probability transform to log pdfs
        [logPfuns,nocell_tr] = trans_logPfuns(logPfuns,trinfo);
        
    else
        trinfo = [];
    end

end

function [logPfuns,nocell_tr] = trans_logPfuns(logPfuns_orig,trinfo)
%TRANS_LOGPFUNS Add log probability transform to log pdfs
    logPfuns = [];
    logPfuns{1} = @(y) pdftrans(y,'logpdf',trinfo);
    if iscell(logPfuns_orig)
        for i = 1:numel(logPfuns_orig)
            logPfuns{i+1} = @(y) logPfuns_orig{i}(pdftrans(y,'inverse',trinfo));
        end
        nocell_tr = 0;
    else
        logPfuns{2} = @(y) logPfuns_orig(pdftrans(y,'inverse',trinfo));
        nocell_tr = 1;
    end
end

function [trinfo,Y] = trans_gauss(X,trinfo,options)
%TRANS_GAUSS Apply nonlinear transformation to Gaussify variables.

    nvars = size(X,2);
    [theta,Y] = reparam(X,trinfo.lb_orig,trinfo.ub_orig, ...
        options.VarTransformMethod,options.VarTransformRestarts);
    switch options.VarTransformMethod
        case 1
            trinfo.type = 5*ones(1,nvars);
            trinfo.alpha = theta(:,1)';
            trinfo.beta = theta(:,2)';
        case 2
            trinfo.alpha = theta(:,1)';
            trinfo.beta = theta(:,2)';
            trinfo.mu = theta(:,3)';
            trinfo.gamma = theta(:,4)';
            trinfo.type = 6*ones(1,nvars);
        case 3
            nparams = size(theta,2);
            trinfo.xspace = bsxfun(@plus,bsxfun(@times,theta,trinfo.ub_orig'-trinfo.lb_orig'),trinfo.lb_orig');
            trinfo.pspace = repmat([0,linspace(1/(nparams-1),1-1/(nparams-1),nparams-2),1],[nvars,1]);
            trinfo.type = 7*ones(1,nvars);
        case 4
            nparams = numel(theta);
            for i = 1:nparams
                %theta{i}.Mu = theta{i}.Mu*(trinfo.ub_orig(i)-trinfo.lb_orig(i))+trinfo.lb_orig(i);
                %theta{i}.Sigma = theta{i}.Sigma*(trinfo.ub_orig(i)-trinfo.lb_orig(i));
            end
            trinfo.gmm = theta;
            trinfo.type = 8*ones(1,nvars);
    end
    trinfo.scale = max(abs(Y),[],1);
    Y = bsxfun(@rdivide,Y,trinfo.scale);
end

%--------------------------------------------------------------------------
function fval = assignfval(fval,ftemp,iSample)
%ASSIGNFVAL Assign values of log fun.
    for j = 1:numel(ftemp)
        if isempty(ftemp{j})
            fval{j}(iSample,:) = NaN;
        else
            fval{j}(iSample,:) = ftemp{j};
        end
    end
end
%--------------------------------------------------------------------------
function [sampleState,logPfuns,nocell] = sampleinit(logPfuns,nvars,N,LB,UB,widths,covmat,options)
%SAMPLEINIT Initialize parameters and options for sampling algorithm.

if ~iscell(logPfuns); logPfuns = {logPfuns}; nocell = 1; else nocell = 0; end

% Try to convert to function handle
Nfuns = numel(logPfuns);
for k = 1:Nfuns
    if ischar(logPfuns{k}); logPfuns{k} = str2func(logPfuns{k}); end
end

if numel(LB) == 1; LB = repmat(LB, [1,nvars]); end
if numel(UB) == 1; UB = repmat(UB, [1,nvars]); end
if size(LB,1) > 1; LB = LB'; end
if size(UB,1) > 1; UB = UB'; end
if numel(widths) == 1; widths = repmat(widths, [1,nvars]); end
% LB_out = LB - eps(LB);
% UB_out = UB + eps(UB);
sampleState.basewidths = widths;    % User-supplied widths

if isempty(widths); widths = (UB - LB)/2; end
widths(isinf(widths)) = 10;
widths(LB == UB) = 1;   % WIDTHS is irrelevant when LB == UB, set to 1
sampleState.widths = widths;

[cholsigma,p] = chol(covmat);
sampleState.cholsigma = cholsigma;

sampleState.funccount = zeros(1,Nfuns);
sampleState.thin = floor(options.Thin);
sampleState.burn = floor(options.Burnin);

% Total number of samples
sampleState.totN = sampleState.burn + N + (N-1)*(sampleState.thin-1);

sampleState.LB = LB;
sampleState.UB = UB;

% Gaussian mixture model approximation
sampleState.gmm = [];

end
%--------------------------------------------------------------------------
function x0 = sampleinitpop(logfun,x0,K,sampleState,options,trace,varargin)
%SAMPLEINITPOP Initialize population in small ball(s) around starting point(s).

if isempty(trace); trace = 0; end

D = size(x0,2);
LB = sampleState.LB;
UB = sampleState.UB;

% Scramble points
x0 = x0(randperm(size(x0,1)),:);

% Only one point requested
if K == 1
    x0 = x0(1,:);
    return;
end

% Duplicate points if not enough
if size(x0,1) < K
    x0 = [x0; repmat(x0,[floor(K/size(x0,1)),1])];
    if trace > 1; fprintf(['Generating ' num2str(K) ' initial starting points...\n']); end
end

x0 = x0(1:K,:); % Take only first K initial points    

% Add small jitter to all points
if isfield(options,'TolX'); tolx = options.TolX; else tolx = 1e-6; end
widths = sampleState.widths;
x0 = bsxfun(@plus,x0,bsxfun(@times,sqrt(tolx)*widths,randn(K,D)));    

% Move starting point within bounds (mirror off bounds)
moved = 0;

% Keep points fixed along fixed dimensions
fixed_dim = LB == UB;
if any(fixed_dim) && any(any(bsxfun(@gt,x0(:,fixed_dim),LB(fixed_dim)))) || ...
    any(any(bsxfun(@lt,x0(:,fixed_dim),LB(fixed_dim))))
        moved = 1;            
        x0(:,fixed_dim) = LB(fixed_dim); 
end

% Reflect points inside bounds
while any(any(bsxfun(@lt, x0, LB))) || any(any(bsxfun(@gt, x0, UB)))
    moved = moved + 1;
    extra = bsxfun(@minus,LB,x0);
    extra(extra < 0) = 0;
    x0 = bsxfun(@plus,x0,2*extra);
    extra = bsxfun(@minus,x0,UB);
    extra(extra < 0) = 0;
    x0 = bsxfun(@minus,x0,2*extra);
end

% Effective bounds for unbounded parameters
effLB = LB;
effLB(isfinite(UB) & LB == -Inf) = UB(isfinite(UB) & LB == -Inf) - 10;
effLB(UB == Inf & LB == -Inf) = -10;
effUB = UB;
effUB(isfinite(LB) & UB == Inf) = LB(isfinite(LB) & UB == Inf) + 10;
effUB(UB == Inf & LB == -Inf) = 10;

randomized = 0;

for i = 1:1e3
    
    % Randomly generate points that are still invalid
    invalid = any(bsxfun(@lt, x0, LB) | bsxfun(@gt, x0, UB),2);

    % Multiple functions, it is assumed that the first function encodes bounds
    for j = 1:numel(logfun)
        for k = find(~invalid)'
            % fvaltemp = feval(logfun{j},x0(k,:),varargin{:});
            fvaltemp = feval(logfun{j},x0(k,:));
            if ~isfinite(fvaltemp); invalid(k) = true; end
        end
    end
    
    if ~any(invalid); break; end    
    randomized = 1;

    correct = bsxfun(@plus, bsxfun(@times, effUB - effLB, rand(K,D)), effLB);
    x0(invalid,:) = correct(invalid,:);
end

if any(invalid)
    error('Could not find valid initial starting points.');    
end

if moved > 0 && trace > 1
    fprintf('Starting points outside bounds were reflected inside the function domain.\n');
end
if randomized > 0 && trace > 1
    fprintf('Invalid starting points were randomly initialized.\n');    
end

end

%--------------------------------------------------------------------------
function sampleargcheck(logfun,x0,y,N,sampleState,warmup,options)
%SAMPLEARGCHECK Check arguments and options for sampling algorithm.
%
% Luigi Acerbi 2016

[K,D] = size(x0);
LB = sampleState.LB;
UB = sampleState.UB;

%assert(size(x0,1) == 1 && size(x0,1) == 1, ...
%    'The initial point X0 needs to be a scalar or row vector.');
Nfuns = numel(logfun);
for k = 1:Nfuns
    assert(isa(logfun{k},'function_handle'), ...
        'LOGFUN needs to be a function handle or function name, or a cell array of function handles or function names.');
end
assert(isscalar(N) && N > 0, ...
    'The number of requested samples N needs to be a positive integer.');
assert(size(LB,1) == 1 && size(UB,1) == 1 && numel(LB) == D && numel(UB) == D, ...
    'LB and UB need to be empty matrices, scalars or row vectors with the same number of columns as X0.');
assert(all(UB >= LB), ...
    'All upper bounds UB need to be equal or greater than lower bounds LB.');
if isfield(sampleState,'widths')
    assert(all(sampleState.widths > 0 & isfinite(sampleState.widths)) ...
        && isreal(sampleState.widths), ...
        'The vector WIDTHS need to be all positive real numbers.');
end
assert(all(all(bsxfun(@ge, x0, LB))) & all(all(bsxfun(@le, x0, UB))), ...
    'Initial starting point X0 outside the bounds.');
if size(x0,1) == 1
    assert(all(isfinite(y)) && isreal(y), ...
        'Initial starting point X0 needs to evaluate to a real number (not Inf or NaN).');    
else
    assert(all(isfinite(y)) && isreal(y), ...
        'All initial starting points X0 need to evaluate to a real number (no Inf or NaN).');
end
assert(sampleState.thin > 0 && isscalar(sampleState.thin), ...
    'The thinning factor OPTIONS.Thin needs to be a positive integer.');
assert(sampleState.burn >= 0 && isscalar(sampleState.burn), ...
    'The burn-in samples OPTIONS.Burnin needs to be a non-negative integer.');

if sampleState.totN + size(warmup,1) < K*20 && options.Diagnostics
    warning(['The total number of samples, including warmup, is likely too small for the number of walkers (K=' num2str(K) '). Increase N or OPTIONS.Thin.']);
end
if K > 1 && K < 2*D
    warning(['The number of walkers for ensemble sampling (K=' num2str(K) ') should be at least twice the dimensionality of the data (D=' num2str(D) ').']);
end

end

%--------------------------------------------------------------------------
function [x,logP,fval,exitflag,sampleState] = transSliceSampleRD(fun,x,logP,fval,sampleState,options,printfun)
%TRANSSLICESAMPLECW Random direction slice sample transition operator.

if nargin < 7; printfun = []; end

D = size(x,2);
LB = sampleState.LB;
UB = sampleState.UB;
jacobianflag = 0;
dd = 0;

if ~isfield(sampleState,'gmm'); sampleState.gmm = []; end
if ~isfield(sampleState,'vbmodel'); sampleState.vbmodel = []; end
if ~isfield(sampleState,'cholsigma'); sampleState.cholsigma = []; end
gmm = sampleState.gmm;
vbmodel = sampleState.vbmodel;
cholsigma = sampleState.cholsigma;

if size(sampleState.others,1) >= 2
    % Parallel slice sampling
    ord = randperm(size(sampleState.others,1));        
    xr = sampleState.others(ord(1:2),:);
    wvec = (xr(2,:) - xr(1,:))*options.SigmaFactor;
    wsize = 1;

elseif ~isempty(vbmodel)
    % Surrogate random-direction slice sampling
    xr = vbgmmrnd(vbmodel,2);
    wvec = (xr(:,2) - xr(:,1))'*options.SigmaFactor;
    wsize = 1;

else
    % Random-direction slice sampling
    wsize = 1;
    uvec = randn(1,D);
    uvec = uvec / norm(uvec);    
    if ~isempty(cholsigma)
        wvec = uvec*cholsigma;      % Covariance-based sampling
    else
        wvec = uvec.*sampleState.widths;        
    end    
end
        
[x,logP,fval,exitflag,fc] = ...
    slicesample1(dd,fun,wvec,x,logP,...
    fval,wsize,jacobianflag,LB,UB,gmm,printfun,options);
sampleState.funccount = sampleState.funccount + fc;

% Slice sampler collapsed on point
if exitflag
    if isfield(sampleState,'slicecollapsed')
        sampleState.slicecollapsed = sampleState.slicecollapsed + 1;
    else
        sampleState.slicecollapsed = 1;
    end
end

end
%--------------------------------------------------------------------------
function [xnew,logPnew,fvalnew,exitflag,funccount] = ...
    slicesample1(dd,fun,wvec,x_c,logP,fval,wsize,jacobianflag,LB,UB,gmm,printfun,options)
%SLICESAMPLE1 Slice sampling along a given direction.
%
% The standard slice sampling code is inspired by a MATLAB implementation 
% of slice sampling by Iain Murray. See pseudo-code in MacKay (2003).

D = size(x_c,2);
logjacobian = 0;
funccount = 0;
debugplot = 0;

% If a GMM is supplied, perform inversion slice sampling
inversion_sample = ~isempty(gmm) && options.InversionSample;

% Sample auxiliary variable
log_uprime = log(rand()) + logP;

x_start = x_c;  % Save starting point

% Try computing slice for inversion slice sampling (might fail)
if inversion_sample
    try
        if ~isfield(gmm,'ischol'); gmm.ischol = 0; end
        if D > 1
            % Multivariate GMM, condition on direction        
            [g.w1,g.mu1,v1] = gmmslice1(wvec,x_c,gmm.w,gmm.Mu,gmm.Sigma,gmm.ischol);
            g.s1 = sqrt(v1);
        else
            g.w1 = gmm.w(:)';
            g.mu1 = gmm.Mu(:)'-x_c;
            if gmm.ischol
                g.s1 = gmm.Sigma(:)';
            else
                g.s1 = sqrt(gmm.Sigma(:)');
            end
        end
        
        if debugplot
            hold off;
            xx = linspace(-5,5,1e4);
            yy = gmm1pdf(xx,g.w1,g.mu1,g.s1);
            plot(xx,yy,'-k','LineWidth',1);
            hold on;
            plot([0 0],[0, max(yy)],'-k','LineWidth',2);
        end
        
    catch
        warning('Inversion slice sampling failed. Trying standard slice sampling.');
        g = [];
    end
end
    
if inversion_sample && ~isempty(g)
    %% Inversion slice sampling    
    if isfield(options,'MinWeightGlobalComponent') ...
            && options.MinWeightGlobalComponent > 0
        wbase = sum(g.w1(1:end-1));
        g.w1(end) = max(options.MinWeightGlobalComponent*wbase, g.w1(end));
        g.w1 = g.w1/sum(g.w1);
    end
        
    g.mu1 = g.mu1/norm(wvec);
    g.s1 = g.s1/norm(wvec);
    
    % Effective dimensionality of adaptive-direction Jacobian transform
    if jacobianflag; g.effjdim = D; else g.effjdim = 1; end
    
    % Convert to cdf variable
    x_c0 = x_c;
    x_c = gmm1djcdf(0,g.w1,g.mu1,g.s1,g.effjdim);
    x_l = 0;
    rr = x_c;
    % tolx = options.TolF;
    tolx = eps;
    
    wvec0 = wvec;
    wvec = 1;
    wsize = 1;
    
    wrappedfun = fun;
    fun = @(x_) inversepdfbnd(x_,x_c0,wvec0,wrappedfun,g,options);
    
    delta = 0;
    lf = log(gmm1djpdf(delta,g.w1,g.mu1,g.s1,g.effjdim));
    log_uprime = log_uprime - lf;
    
    if debugplot
        hold off;
        xx = linspace(-3,3,1e3);
        yy = zeros(size(xx));
        for i = 1:numel(yy); yy(i) = gmm1djpdf(xx(i),g.w1,g.mu1,g.s1,g.effjdim); end
        plot(xx,yy,'k-','LineWidth',1);
        box off;
        set(gca,'TickDir','out');
        set(gcf,'Color','w');
        drawnow;
    end
    
else
    %% Standard slice sampling
    
    % Create interval (x_l, x_r) of size wsize (in wvec units) enclosing x_c
    rr = rand*wsize;    % rr is position of x_c with respect to x_l
    x_l = x_c - rr*wvec;
    x_r = x_c + (wsize-rr)*wvec;

    % Adjust interval to outside bounds for bounded problems (needs checking)
    if any(isfinite(LB) | isfinite(UB))
        if any(x_l < LB) || any(x_l > UB)
            delta = (LB - x_l).*(x_l < LB) + (x_l - UB).*(x_l > UB);
            proj = delta./abs(wvec);
            shift_l = max(proj);
            x_l = x_l + wvec*shift_l;
            rr = rr - shift_l;
        end

        if any(x_r > UB) || any(x_r < LB)
            delta = (x_r - UB).*(x_r > UB) + (LB - x_r).*(x_r < LB);
            proj = delta./abs(wvec); % *(wvec/sqrt(wvec*wvec'));
            shift_r = max(proj);
            x_r = x_r-shift_r;
        end
    end

    % [x_l; x_r]
    % x_l + rr*wvec - x_c

    % Step-out procedure
    if options.StepOut
        error('STEPOUT needs to be checked.');
        
        steps = 0;
        wbase = wsize;

        while 1
            [logtemp,~,~,fc] = fun(x_l);
            funccount = funccount + fc;
            if jacobianflag; logjacobian = (D-1)*log(abs(1-(-rr))); end
            if (logtemp + logjacobian) <= log_uprime; break; end
            x_l = x_l - wbase*wvec;
            wsize = wsize + wbase;
            rr = rr + wbase;
            steps = steps + 1;
        end
        while 1
            [logtemp,~,~,fc] = fun(x_r);
            funccount = funccount + fc;
            if jacobianflag; logjacobian = (D-1)*log(abs(1-(wsize-rr))); end
            if (logtemp + logjacobian) <= log_uprime; break; end
            x_r = x_r + wbase*wvec;
            wsize = wsize + wbase;
            rr = rr + wbase;
            steps = steps + 1;
        end

        if ~isempty(printfun) && steps >= 10
            if dd == 0
                action = ['step-out adaptive dim (' num2str(steps) ' steps)'];
            else        
                action = ['step-out dim ' num2str(dd) ' (' num2str(steps) ' steps)'];
            end
            printfun(action);
        end    
    end
    
    tolx = sqrt(sum((options.TolX.*wvec).^2));
    
end

% Shrink procedure: propose XNEW and shrink interval until good one found
shrink = 0;
exitflag = 0;

% xx = linspace(0,1,101);
% for i = 1:numel(xx); yy(i) = fun(x_l + xx(i)*wsize*wvec); end
% yy = exp(yy-max(yy)); hold off;
% plot(xx,yy,'k-','LineWidth',1); hold on;
% plot(rr/wsize*[1 1],[0,max(yy)],'k--','LineWidth',1);
% pause;

while 1
    shrink = shrink + 1;
    tolr = tolx/wsize;
    
    rr2 = rand()*wsize;
    xnew = x_l + rr2*wvec;
    
    if inversion_sample && jacobianflag
        [logPnew,fvalnew,fc,delta] = fun(xnew);
    else
        [logPnew,fvalnew,fc] = fun(xnew);
    end
    funccount = funccount + fc;
    
    if jacobianflag
        if inversion_sample
            logjacobian = (D-1)*log(abs(1-delta));            
        else
            logjacobian = (D-1)*log(abs(1-(rr2-rr)));
        end
    end
    
    if logPnew + logjacobian > log_uprime
        break;          % This is the main way to leave the while loop
    else
        % Shrink in
        if rr2 > (rr + tolr)
            wsize = rr2;
        elseif rr2 < (rr - tolr)
            x_l = xnew;
            rr = rr - rr2;
            wsize = wsize - rr2;
        else    % Shrunk too close to current point
            exitflag = 1;
            if options.Noise
                if ~isempty(printfun)
                    printfun(['shrunk to point (' num2str(shrink) ' steps)']);
                    printfun = [];
                end
            else
                errorstr = ['Shrunk to current position and proposal still not acceptable. ' ...
                'Current position: ' num2str(xnew,' %g') '. ' ...
                'Log f: (new value) ' num2str(logPnew), ', (target value) ' num2str(log_uprime) '.'];
                warning(errorstr);                
            end
            % Reset to current point
            xnew = x_start;
            logPnew = logP;
            fvalnew = fval;
            break;  % Shrunk too close - bad way to leave the while loop
        end
    end
end

% shrink

% Inversion sampling - transform back to pdf coordinates (if not collapsed)
if inversion_sample && exitflag == 0
    delta = gmm1djinv(xnew,g.w1,g.mu1,g.s1,g.effjdim,options.TolX);
    xnew = x_c0 + wvec0*delta;
    lf = log(gmm1djpdf(delta,g.w1,g.mu1,g.s1,g.effjdim));
    logPnew = logPnew + lf;
end

if ~isempty(printfun) && shrink >= 10
    if dd == 0
        action = ['shrink adaptive dim (' num2str(shrink) ' steps)'];        
    else
        action = ['shrink dim ' num2str(dd) ' (' num2str(shrink) ' steps)'];
    end
    printfun(action);
end

end

%--------------------------------------------------------------------------
function [logP,fval,funccount,delta] = inversepdfbnd(p,x_c0,wvec0,wrappedfun,g,options)
%INVERSEPDFBND Compute pdf for inversion sampling.
    
% Outside cdf bounds, return immediately
if p <= 0 || p >= 1
    logP = -Inf;
    fval = [];
    funccount = 0;
    return;
end

w1 = g.w1;
mu1 = g.mu1;
s1 = g.s1;
effjdim = g.effjdim;

% [delta,fval,exitflag,output] = gmm1djinv(x,w1,mu1,s1,effjdim,options.TolX);
% Map from cdf space to pdf space
delta = gmm1djinv(p,w1,mu1,s1,effjdim,options.TolX);        
xtemp = x_c0 + wvec0*delta;
[logP,fval,funccount] = wrappedfun(xtemp);
if isfinite(logP)
    lf = log(gmm1djpdf(delta,w1,mu1,s1,effjdim));
    logP = logP - lf;
end

end

%--------------------------------------------------------------------------
function [logP,fval,funccount] = logpdfbnd(x,logfun,LB,UB,options,varargin)
%LOGPDFBND Evaluate log pdf with bounds and prior.

% Default OPTIONS
if nargin < 5 || isempty(options)
    options.Temperature = 1;
end

Nfuns = numel(logfun);
funccount = zeros(1,Nfuns);
for k = 1:Nfuns; fval{k} = []; end

if any(x < LB | x > UB)
    logP = -Inf;
    return;
end

for k = 1:Nfuns
    fval{k} = feval(logfun{k},x,varargin{:});
    funccount(k) = funccount(k) + 1;
    if any(~isfinite(fval{k}))
        logP = -Inf;
        if any(isnan(fval{k}))
            warning(['Log density #' num2str(k) ' returned NaN. Trying to continue.']);
        elseif any(fval{k} == Inf)
            warning(['Log density #' num2str(k) ' returned Inf. Trying to continue.']);
        end
        return;        
    end
end

logP = sum(cellfun(@sum,fval));

end

%--------------------------------------------------------------------------
function y = cellgetrow(x,n)
%CELLGETROW Get the N-th row of each matrix in cell array X.

y = cellfun(@(x_) getrow(x_,n),x,'UniformOutput',0);

    function y = getrow(x,n)
        y = x(n,:);
    end

end

