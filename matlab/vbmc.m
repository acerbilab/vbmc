function [vp,elbo,elbo_sd,exitflag,output,stats] = vbmc(fun,x0,LB,UB,PLB,PUB,options,varargin)
%VBMC Posterior and model inference via Variational Bayesian Monte Carlo (v0.7)
%   Documentation to be written -- work in progress.
% 

%--------------------------------------------------------------------------
% VBMC: Variational Bayesian Monte Carlo for posterior and model inference.
% To be used under the terms of the GNU General Public License 
% (http://www.gnu.org/copyleft/gpl.html).
%
%   Author (copyright): Luigi Acerbi, 2018
%   e-mail: luigi.acerbi@{gmail.com,nyu.edu,unige.ch}
%   URL: http://luigiacerbi.com
%   Version: 0.7 (alpha)
%   Release date: May 2, 2018
%   Code repository: https://github.com/lacerbi/vbmc
%--------------------------------------------------------------------------

% TO-DO list:
% - Write a private quantile function to avoid calls to Stats Toolbox.
% - Fix call to fmincon if Optimization Toolbox is not available.
% - Check that I am not using other ToolBoxes by mistake.



%% Basic default options

defopts.Display                 = 'iter         % Level of display ("iter", "notify", "final", or "off")';
defopts.MaxIter                 = '20*nvars     % Max number of iterations';
defopts.MaxFunEvals             = '200*nvars    % Max number of objective fcn evaluations';
defopts.NonlinearScaling        = 'on           % Automatic nonlinear rescaling of variables';
defopts.OutputFcn               = '[]           % Output function'; 
defopts.UncertaintyHandling     = '[]           % Explicit noise handling (if empty, determine at runtime)';
defopts.NoiseSize               = '[]           % Base observation noise magnitude';
%defopts.NoiseFinalSamples       = '10           % Samples to estimate FVAL at the end (for noisy objectives)';
defopts.Fvals                   = '[]           % Evaluated fcn values at X0';
defopts.OptimToolbox            = '[]           % Use Optimization Toolbox (if empty, determine at runtime)';
defopts.Diagnostics             = 'on           % Run in diagnostics mode, get additional info';

%% If called with no arguments or with 'defaults', return default options
if nargout <= 1 && (nargin == 0 || (nargin == 1 && ischar(fun) && strcmpi(fun,'defaults')))
    if nargin < 1
        fprintf('Basic default options returned (type "help vbmc" for help).\n');
    end
    vp = defopts;
    return;
end

%% Advanced options (do not modify unless you *know* what you are doing)
defopts.FunEvalStart       = 'max(D,10)         % Number of initial objective fcn evals';
defopts.FunEvalsPerIter    = '10                % Number of objective fcn evals per iteration';
defopts.AcqFcn             = '@vbmc_acqkl       % Expensive acquisition fcn';
% defopts.AcqFcn             = '@vbmc_acqkl       % Expensive acquisition fcn';
defopts.Nacq               = '20                % Expensive acquisition fcn evals per new point';
defopts.NSsearch           = '2048              % Samples for fast acquisition fcn eval per new point';
defopts.NSent              = '100               % Samples per component for fast Monte Carlo approx. of the entropy';
defopts.NSentFine          = '2^15              % Samples per component for refined Monte Carlo approx. of the entropy';
defopts.NSelbo             = '50                % Samples per component for fast approx. of ELBO';
defopts.ElboStarts         = '2                 % Starting points to refine optimization of the ELBO';
defopts.NSgpMax            = '80                % Max GP hyperparameter samples (decreases with training points)';
defopts.StopGPSampling     = '200 + 10*nvars    % Stop GP hyperparameter sampling (start optimizing)';
defopts.TolGPVar           = '1e-4              % Stop GP hyperparameter sampling if sample variance is below this threshold per fcn';
defopts.QuadraticMean      = 'yes               % Use GP with quadratic mean function (otherwise constant)';
defopts.Kfun               = '@sqrt             % Variational components as a function of training points';
defopts.HPDFrac            = '0.5               % High Posterior Density region (fraction of training inputs)';
defopts.WarpRotoScaling    = 'on                % Rotate and scale input';
%defopts.WarpCovReg         = '@(N) 25/N         % Regularization weight towards diagonal covariance matrix for N training inputs';
defopts.WarpCovReg         = '0                 % Regularization weight towards diagonal covariance matrix for N training inputs';
defopts.WarpNonlinear      = 'on                % Nonlinear input warping';
defopts.WarpEpoch          = '20                % Recalculate warpings after this number of fcn evals';
defopts.ELCBOWeight        = '1                 % Uncertainty weight during ELCBO optimization';
defopts.TolLength          = '1e-6              % Minimum fractional length scale';
defopts.NoiseObj           = 'off               % Objective fcn returns noise estimate as 2nd argument (unsupported)';
defopts.CacheSize          = '1e4               % Size of cache for storing fcn evaluations';
defopts.CacheFrac          = '0.5               % Fraction of search points from starting cache (if nonempty)';
defopts.TolFunAdam         = '0.001             % Stopping threshold for Adam optimizer';
defopts.TolSD              = '0.1               % Tolerance on ELBO uncertainty for stopping (iff variational posterior is stable)';
defopts.TolsKL             = '0.01*sqrt(nvars)  % Stopping threshold on change of variational posterior per training point';
defopts.TolStableIters     = '4                 % Number of stable iterations for checking stopping criteria';
defopts.TolStableFunEvals  = '4*nvars           % Number of stable fcn evals for checking stopping criteria';
defopts.KLgauss            = 'yes               % Use Gaussian approximation for symmetrized KL-divergence b\w iters';
defopts.TrueMean           = '[]                % True mean of the target density (for debugging)';
defopts.TrueCov            = '[]                % True covariance of the target density (for debugging)';
defopts.MinFunEvals        = '2*nvars^2         % Min number of fcn evals';
defopts.MinIter            = 'nvars             % Min number of iterations';
defopts.HeavyTailSearchFrac = '0.5               % Fraction of search points from heavy-tailed variational posterior';
defopts.SearchSampleGP     = 'false              % Generate search candidates sampling from GP surrogate';

%% If called with 'all', return all default options
if strcmpi(fun,'all')
    vp = defopts;
    return;
end

%% Check that all VBMC subfolders are on the MATLAB path
add2path();

%% Input arguments

if nargin < 3 || isempty(LB); LB = -Inf; end
if nargin < 4 || isempty(UB); UB = Inf; end
if nargin < 5; PLB = []; end
if nargin < 6; PUB = []; end
if nargin < 7; options = []; end

%% Initialize display printing options

if ~isfield(options,'Display') || isempty(options.Display)
    options.Display = defopts.Display;
end

switch lower(options.Display(1:3))
    case {'not','notify','notify-detailed'}
        prnt = 1;
    case {'non','none','off'}
        prnt = 0;
    case {'ite','all','iter','iter-detailed'}
        prnt = 3;
    case {'fin','final','final-detailed'}
        prnt = 2;
    otherwise
        prnt = 1;
end

%% Initialize variables and algorithm structures

if isempty(x0)
    if prnt > 2
        fprintf('X0 not specified. Taking the number of dimensions from PLB and PUB...');
    end
    if isempty(PLB) || isempty(PUB)
        error('vbmc:UnknownDims', ...
            'If no starting point is provided, PLB and PUB need to be specified.');
    end    
    x0 = NaN(size(PLB));
    if prnt > 2
        fprintf(' D = %d.\n', numel(x0));
    end
end

D = size(x0,2);     % Number of variables
optimState = [];

% Check correctness of boundaries and starting points
[LB,UB,PLB,PUB] = boundscheck(x0,LB,UB,PLB,PUB,prnt);

% Convert from char to function handles
if ischar(fun); fun = str2func(fun); end

% Setup algorithm options
[options,cmaes_opts] = setupoptions(D,defopts,options);

% Setup and transform variables
K = getK(options.FunEvalStart,options.Kfun);
[vp,optimState] = ...
    setupvars(x0,LB,UB,PLB,PUB,K,optimState,options,prnt);

% Store objective function
optimState.fun = fun;
if isempty(varargin)
    funwrapper = fun;   % No additional function arguments passed
else
    funwrapper = @(u_) fun(u_,varargin{:});
end

% Initialize function logger
[~,optimState] = vbmc_funlogger([],x0(1,:),optimState,'init',options.CacheSize,options.NoiseObj);

% GP hyperparameters
hyp = [];   hyp_warp = [];  gp = [];
if options.QuadraticMean
    optimState.gpMeanfun = 'negquad';
else
    optimState.gpMeanfun = 'const';
end

if optimState.Cache.active
    displayFormat = ' %5.0f     %5.0f  /%5.0f   %12.3g  %12.3g  %12.3g     %4.0f %10.3g       %s\n';
else
    displayFormat = ' %5.0f       %5.0f    %12.3g  %12.3g  %12.3g     %4.0f %10.3g     %s\n';
end
if prnt > 2
    if optimState.Cache.active
        fprintf(' Iteration f-count/f-cache    Mean[ELBO]     Std[ELBO]     sKL-iter[q]   K[q]  Convergence    Action\n');
        % fprintf(displayFormat,0,0,0,NaN,NaN,NaN,NaN,Inf,'');        
    else
        fprintf(' Iteration   f-count     Mean[ELBO]     Std[ELBO]     sKL-iter[q]   K[q]  Convergence  Action\n');
        % fprintf(displayFormat,0,0,NaN,NaN,NaN,NaN,Inf,'');        
    end
end

%% Variational optimization loop
iter = 0;
isFinished_flag = false;
exitflag = 0;   output = [];    stats = [];

while ~isFinished_flag    
    iter = iter + 1;
    optimState.iter = iter;
    vp_old = vp;
    action = '';
    
    %% Adaptively sample new points into the training set
    optimState.trinfo = vp.trinfo;
    if iter == 1; new_funevals = options.FunEvalStart; else; new_funevals = options.FunEvalsPerIter; end
    [optimState,t_adapt(iter),t_func(iter)] = ...
        adaptive_sampling(optimState,new_funevals,funwrapper,vp,vp_old,gp,options);
    optimState.N = optimState.Xmax;  % Number of training inputs

    timer_fits = tic;
        
    % Warping iteration?
    isWarping = (optimState.N - optimState.LastWarping) >= options.WarpEpoch ...
        && (options.MaxFunEvals - optimState.N) >= options.WarpEpoch;
    if isWarping
        optimState.LastWarping = optimState.N;
        if isempty(action); action = 'warp'; else; action = [action ', warp']; end
    end
    
    %% Update stretching of unbounded variables
    if any(isinf(LB) & isinf(UB)) && (isWarping || iter == 1)
        [vp,optimState,hyp] = ...
            warp_unbounded(vp,optimState,hyp,gp,cmaes_opts,options);
    end
    
    %%  Rotate and rescale variables
    if options.WarpRotoScaling && isWarping
        [vp,optimState,hyp] = ...
            warp_rotoscaling(vp,optimState,hyp,gp,cmaes_opts,options);
    end
    
    %% Learn nonlinear warping via GP
    if options.WarpNonlinear && isWarping
        [vp,optimState,hyp,hyp_warp] = ...
            warp_nonlinear(vp,optimState,hyp,hyp_warp,cmaes_opts,options);
    end
        
    %% Train GP in warped space
    
    % Check whether to perform hyperparameter sampling or optimization
    if optimState.StopSampling == 0
        % Number of samples
        Ns_gp = round(options.NSgpMax/sqrt(optimState.N));
        
        % Stop sampling after reaching max number of training points
        if optimState.N >= options.StopGPSampling
            optimState.StopSampling = optimState.N;
        end
        
        % Stop sampling after sample variance stays below Tol for a while
        [idx_stable,dN,dN_last] = getStableIter(stats,optimState,options);
        if ~isempty(idx_stable) && idx_stable > 1
            varss_list = stats.gpSampleVar;
            if sum(varss_list(idx_stable-1:iter-1)) < options.TolGPVar*dN && ...
                    varss_list(end) < options.TolGPVar*dN_last
                optimState.StopSampling = optimState.N;
            end
        end
        
        if optimState.StopSampling > 0
            if isempty(action); action = 'gp2opt'; else; action = [action ', gp2opt']; end
        end
    end
    if optimState.StopSampling > 0; Ns_gp = 0; end
    
    % Get priors and starting hyperparameters
    [hypprior,X_hpd,y_hpd,~,hyp0,optimState.gpMeanfun] = ...
        vbmc_gphyp(optimState,optimState.gpMeanfun,0,options);
    if isempty(hyp); hyp = hyp0; end % Initial GP hyperparameters
    gptrain_options.Nopts = 3;
    
    [gp,hyp] = gplite_train(hyp,Ns_gp, ...
        optimState.X(1:optimState.Xmax,:),optimState.y(1:optimState.Xmax), ...
        optimState.gpMeanfun,hypprior,[],gptrain_options);
    
    if 1
        if D == 1
            Xs = linspace(min(optimState.X(1:optimState.Xmax,:)),max(optimState.X(1:optimState.Xmax,:)),3e3)';    
            [ymu,ys2,fmu,fs2] = gplite_pred(gp,Xs);
            ymu = mean(ymu,2); ys2 = mean(ys2,2);
            plot(Xs,ymu,'k-','LineWidth',1); hold on;
            plot(Xs,ymu+sqrt(ys2),'k:','LineWidth',1);
            plot(Xs,ymu-sqrt(ys2),'k:','LineWidth',1);
            scatter(optimState.X(1:optimState.Xmax,:),optimState.y(1:optimState.Xmax),'ro');  hold off;
            axis([-5 5 -10 3]);        
            drawnow
        elseif D == 2 && 0
            subplot(2,2,4);
            scatter(optimState.X(1:optimState.Xmax,1),optimState.X(1:optimState.Xmax,2),'k.'); hold on;
            axis([-5 5 -5 5]);        
            drawnow;
        end
    end
    
    % Sample from GP
    if ~isempty(gp) && 0
        Xgp = vbmc_gpsample(gp,1e3,optimState,1);
        cornerplot(Xgp);
    end
    
    
    %% Optimize variational parameters    
    Knew = getK(optimState.N,options.Kfun);

    % Get bounds for variational parameters optimization    
    [vp.LB_theta,vp.UB_theta] = vbmc_vpbnd(vp,X_hpd,Knew,options);
        
    Nopts = options.NSelbo * Knew; % Number of initial starting points
    nelcbo_fill = zeros(Nopts,1);
        
    nelbo = zeros(options.ElboStarts*2+1,1);     nelcbo = nelbo;
    varF = nelbo; G = varF; H = varF; varss = varF;
    theta_new = NaN(options.ElboStarts*2+1,numel(vp.LB_theta));

    % Generate random initial starting point for variational parameters
    [vp0_vec,vp0_type] = vbinitrnd(Nopts,vp,Knew,X_hpd,y_hpd);

    compute_var = 2;    % Use diagonal-only approximation
    % Confidence weight
    elcbo_beta = options.ELCBOWeight; % * sqrt(vp.D) / sqrt(optimState.N);
    
    % Quickly estimate ELCBO at each candidate variational posterior
    for iOpt = 1:Nopts
        [theta0,vp0_vec(iOpt)] = ...
            get_theta(vp0_vec(iOpt),vp.LB_theta,vp.UB_theta,vp.optimize_lambda);        
        [nelbo_tmp,~,~,~,varF_tmp] = vbmc_negelcbo(theta0,0,vp0_vec(iOpt),gp,options.NSent,0,compute_var);
        nelcbo_fill(iOpt) = nelbo_tmp + elcbo_beta*sqrt(varF_tmp);
    end
    
    % Sort by negative ELCBO
    [~,vp0_ord] = sort(nelcbo_fill,'ascend');
    vp0_vec = vp0_vec(vp0_ord);
    vp0_type = vp0_type(vp0_ord);
    
    for iOpt = 1:options.ElboStarts        
        iOpt_start = iOpt*2-1;
        iOpt_end = iOpt*2;
        
        switch options.ElboStarts
            case 1; idx = 1;
            case 2; if iOpt == 1; idx = find(vp0_type == 1,1); else; idx = find(vp0_type == 2 | vp0_type == 3,1); end
            otherwise; idx = find(vp0_type == (mod(iOpt-1,3)+1),1);
        end
        
        % idx
        
        vp0 = rescale_params(vp0_vec(idx));
        vp0_type(idx) = []; vp0_vec(idx) = [];
        
        theta0 = [vp0.mu(:); log(vp0.sigma(:))];
        if vp.optimize_lambda; theta0 = [theta0; log(vp0.lambda(:))]; end
        theta0 = min(vp.UB_theta',max(vp.LB_theta', theta0));
        
        vbtrainmc_fun = @(theta_) vbmc_negelcbo(theta_,elcbo_beta,vp0,gp,options.NSent,1,compute_var);
        

        % First, fast optimization via entropy approximation
        vbtrain_options = optimoptions('fmincon','GradObj','on','Display','off','OptimalityTolerance',1e-3);
        vbtrain_fun = @(theta_) vbmc_negelcbo(theta_,elcbo_beta,vp0,gp,0,1,compute_var);
        [theta_new(iOpt_end,:),~,~,output] = ...
            fmincon(vbtrain_fun,theta0(:)',[],[],[],[],vp.LB_theta,vp.UB_theta,[],vbtrain_options);
        % output, % pause

        % Second, refine with unbiased stochastic entropy approximation
        theta_new(iOpt_end,:) = ...
            fminadam(vbtrainmc_fun,theta_new(iOpt_end,:),vp.LB_theta,vp.UB_theta,options.TolFunAdam);

        % Recompute ELCBO at start point and endpoint with full variance and more precision
        theta_new(iOpt_start,:) = theta0(:)';
        [nelbo(iOpt_start),~,G(iOpt_start),H(iOpt_start),varF(iOpt_start),~,varss(iOpt_start)] = ...
            vbmc_negelcbo(theta_new(iOpt_start,:),0,vp0,gp,options.NSentFine,0,1);
        nelcbo(iOpt_start) = nelbo(iOpt_start) + elcbo_beta*sqrt(varF(iOpt_start));
        
        [nelbo(iOpt_end),~,G(iOpt_end),H(iOpt_end),varF(iOpt_end),~,varss(iOpt_end)] = ...
            vbmc_negelcbo(theta_new(iOpt_end,:),0,vp0,gp,options.NSentFine,0,1);
        nelcbo(iOpt_end) = nelbo(iOpt_end) + elcbo_beta*sqrt(varF(iOpt_end));
                
        if 1
            mu = reshape(theta_new(iOpt_end,1:D*Knew),[D,Knew]);
            sigma = exp(theta_new(iOpt_end,D*Knew+(1:Knew)));
            if vp.optimize_lambda
                lambda = exp(theta_new(iOpt_end,D*Knew+Knew+(1:D)));
            end

            if D == 1
                hold on;
                Xs = linspace(min(optimState.X(1:optimState.Xmax,:)),max(optimState.X(1:optimState.Xmax,:)),3e3)';
                p = zeros(size(Xs));
                for k = 1:vp.K
                    p = p + normpdf(Xs,mu(k),sigma(k))/vp.K;
                end
                plot(Xs,log(p),'b--','LineWidth',2);
                hold off;
                axis([-5 5 -10 3]);
                drawnow
            elseif D == 2 && 0
                hold on;
                subplot(2,2,4);
                scatter(mu(1,:),mu(2,:),'ro');
                axis([-5 5 -5 5]);
                hold off;
                drawnow;                
            end
            
            %mu
            %sigma
            %if vp.optimize_lambda; lambda, end
        end
        
        vp0_fine(iOpt_start) = vp0;
        vp0_fine(iOpt_end) = vp0;
        
        % fprintf('.'); 
        
        % [nelbo,nelcbo,sqrt(varF),G,H]
    end
    
    % Finally, add variational parameters from previous iteration
    vp0_fine(options.ElboStarts*2+1) = vp;
    [theta0,vp0_fine(options.ElboStarts*2+1)] = get_theta(vp0_fine(options.ElboStarts*2+1),[],[],vp.optimize_lambda);
    theta_new(end,1:numel(theta0)) = theta0;
    [nelbo(end),~,G(end),H(end),varF(end),~,varss(end)] = ...
        vbmc_negelcbo(theta_new(end,1:numel(theta0)),0,vp0_fine(options.ElboStarts*2+1),gp,options.NSentFine,0,1);        
    nelcbo(end) = nelbo(end) + elcbo_beta*sqrt(varF(end));        
        
    % fprintf('\n');
    
    % Take variational parameters with best ELCBO
    [~,idx] = min(nelcbo);
    elbo = -nelbo(idx);
    elbo_sd = sqrt(varF(idx));
    varss = varss(idx);
    % ent = H(idx);
    vp = vp0_fine(idx);
    vp = rescale_params(vp,theta_new(idx,:));
    
    % Compute symmetrized KL-divergence between old and new posteriors
    Nkl = 1e5;
    sKL = max(0,0.5*sum(vbmc_kldiv(vp,vp_old,Nkl,options.KLgauss,1)));
    
    % Compare variational posterior's moments with ground truth
    if ~isempty(options.TrueMean) && ~isempty(options.TrueCov)
        [mubar,Sigma] = vbmc_moments(vp);
        [kl(1),kl(2)] = mvnkl(mubar,Sigma,options.TrueMean,options.TrueCov);
        0.5*sum(kl)
    end

    t_fits(iter) = toc(timer_fits);
    
    dt = (t_adapt(iter)+t_fits(iter))/new_funevals;
    
    if 1
        if D == 1
            hold on;
            Xs = linspace(min(optimState.X(1:optimState.Xmax,:)),max(optimState.X(1:optimState.Xmax,:)),3e3)';
            p = zeros(size(Xs));
            for k = 1:vp.K
                p = p + normpdf(Xs,vp.mu(k),vp.sigma(k))/vp.K;
            end
            plot(Xs,log(p),'b--','LineWidth',2);
            hold off;
            axis([-5 5 -10 3]);
            drawnow
        elseif D == 2 && 0
            hold on;
            x1 = linspace(-5,5,101);
            x2 = linspace(-5,5,101)';
            xx = combvec(x1,x2')';
            yy = vbmc_pdf(xx,vp);
            subplot(2,2,3);
            yy3 = gplite_pred(gp,xx);
            yy3 = exp(yy3);
            hold off;
            surf(x1,x2,reshape(yy3,[numel(x1),numel(x2)]),'EdgeColor','none'); view(0,90)
            axis([-5 5 -5 5]);
            subplot(2,2,2);
            yy2 = zeros(size(xx,1),1);
            for ii = 1:size(xx,1); yy2(ii) = fun(xx(ii,:)); end
            yy2 = exp(yy2);
            surf(x1,x2,reshape(yy2,[numel(x1),numel(x2)]),'EdgeColor','none'); view(0,90); hold on;
            subplot(2,2,1);
            surf(x1,x2,reshape(yy,[numel(x1),numel(x2)]),'EdgeColor','none'); view(0,90); hold on;
            scatter(vp.mu(1,:),vp.mu(2,:),'bo','MarkerFaceColor','b');
            axis([-5 5 -5 5]);
            hold off;
            drawnow;
            % pause
        else
            %xx_warp = vbmc_rnd(3e4,vp,0,1);
            %cornerplot(xx_warp,[],[],[]);
             
            xx = vbmc_rnd(3e4,vp,1,1);
            try
                cornerplot(xx,[],[],[]);
            catch
                % pause
            end
%             vp.mu
%             vp.sigma
%             vp.lambda
        end
        
        %vp
        %vp.lambda        
        
    end    
    
    %mubar
    %Sigma
        
    %----------------------------------------------------------------------
    %% Finalize iteration

    % Record all useful stats
    stats = savestats(stats,optimState,vp,elbo,elbo_sd,varss,sKL,gp,Ns_gp,options.Diagnostics);
        
    % Check termination conditions    
        
    % Maximum number of new function evaluations
    if optimState.funccount >= options.MaxFunEvals
        isFinished_flag = true;
        exitflag = 1;
        % msg = 'Optimization terminated: reached maximum number of function evaluations OPTIONS.MaxFunEvals.';
    end

    % Maximum number of iterations
    if iter >= options.MaxIter
        isFinished_flag = true;
        exitflag = 1;
        % msg = 'Optimization terminated: reached maximum number of iterations OPTIONS.MaxIter.';
    end

    % Reached stable variational posterior with stable ELBO and low uncertainty
    [idx_stable,dN,dN_last] = getStableIter(stats,optimState,options);
    if ~isempty(idx_stable)
        sKL_list = stats.sKL;
        elbo_list = stats.elbo;
        err2 = sum((elbo_list(idx_stable:iter) - mean(elbo_list(idx_stable:iter))).^2);
        qindex(1) = sqrt(err2 / (options.TolSD^2*dN));
        qindex(2) = stats.elboSD(iter) / options.TolSD;
        qindex(3) = sum(sKL_list(idx_stable:iter)) / (options.TolsKL*dN);
        qindex(4) = sKL_list(iter) / (options.TolsKL*dN_last);        
        if all(qindex < 1)
            isFinished_flag = true;
            exitflag = 0;
                % msg = 'Optimization terminated: reached maximum number of iterations OPTIONS.MaxIter.';
        end
        qindex = mean(qindex);
        stats.qindex(iter) = qindex;
    else
        qindex = Inf;
    end
    
    % Prevent early termination
    if optimState.N < options.MinFunEvals || optimState.iter < options.MinIter
        isFinished_flag = false;
    end
    
    % Write iteration
    if optimState.Cache.active
        fprintf(displayFormat,iter,optimState.funccount,optimState.cachecount,elbo,elbo_sd,sKL,vp.K,qindex,action);
    else
        fprintf(displayFormat,iter,optimState.funccount,elbo,elbo_sd,sKL,vp.K,qindex,action);
    end    
    
end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function stats = savestats(stats,optimState,vp,elbo,elbo_sd,varss,sKL,gp,Ns_gp,debugflag)

iter = optimState.iter;
stats.iter(iter) = iter;
stats.N(iter) = optimState.N;
stats.funccount(iter) = optimState.funccount;
stats.cachecount(iter) = optimState.cachecount;
stats.vpK(iter) = vp.K;
stats.elbo(iter) = elbo;
stats.elboSD(iter) = elbo_sd;
stats.sKL(iter) = sKL;
stats.gpSampleVar(iter) = varss;
stats.gpNsamples(iter) = Ns_gp;

if debugflag
    stats.vp(iter) = vp;
    stats.gp(iter) = gp;
end

end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function K = getK(N,Kfun)
%GETK Get number of variational components.

if isnumeric(Kfun)
    K = Kfun;
elseif isa(Kfun,'function_handle')
    K = Kfun(N);
end
K = min(N,max(1,round(K)));
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function add2path()
%ADD2PATH Adds VBMC subfolders to MATLAB path.

% subfolders = {'acq','gpdef','gpml_fast','init','poll','search','utils','warp','gpml-matlab-v3.6-2015-07-07'};
subfolders = {'acq','gplite','misc','utils','warp'};
pathCell = regexp(path, pathsep, 'split');
baseFolder = fileparts(mfilename('fullpath'));

onPath = true;
for iFolder = 1:numel(subfolders)
    folder = [baseFolder,filesep,subfolders{iFolder}];    
    if ispc  % Windows is not case-sensitive
      onPath = onPath & any(strcmpi(folder, pathCell));
    else
      onPath = onPath & any(strcmp(folder, pathCell));
    end
end

% ADDPATH is slow, call it only if folders are not on path
if ~onPath
    addpath(genpath(fileparts(mfilename('fullpath'))));
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [idx_stable,dN,dN_last] = getStableIter(stats,optimState,options)
%GETSTABLEITER Find index of starting stable iteration.

iter = optimState.iter;
idx_stable = [];
dN = [];    dN_last = [];

if ~isempty(stats)
    iter_list = stats.iter;
    N_list = stats.N;
    idx_stable = find(N_list <= optimState.N - options.TolStableFunEvals & ...
        iter_list <= iter - options.TolStableIters,1,'last');
    if ~isempty(idx_stable)
        dN = optimState.N - N_list(idx_stable);
        dN_last = N_list(end) - N_list(end-1);
    end
end

end
