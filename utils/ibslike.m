function [nlogL,nlogLvar,exitflag,output] = ibslike(fun,params,respMat,designMat,options,varargin)
%IBSLIKE Unbiased negative log-likelihood via inverse binomial sampling.
%   NLOGL = IBLLIKE(FUN,PARAMS,RESPMAT,DESIGNMAT) returns unbiased estimate 
%   NLOGL of the negative of the log-likelihood for the simulated model 
%   and data, calculated using inverse binomial sampling (IBS). 
%   FUN is a function handle to a function that simulates the model's 
%   responses (see below). 
%   PARAMS is the parameter vector used to simulate the model's responses. 
%   RESPMAT is a "response" data matrix, where each row correspond to one 
%   observation or "trial" (e.g., a trial of a psychophysical experiment), 
%   and each column represents a different response feature (e.g., the 
%   subject's response and reported confidence level). Responses need to 
%   belong to a finite set.
%   DESIGNMAT is an optional experimental design matrix, where each row 
%   corresponds to one trial, and each column corresponds to a different 
%   trial feature (such as condition, stimulus value, etc.).
%
%   FUN takes as input a vector of parameters PARAMS and an experimental 
%   design matrix DMAT (one row per trial), and generates a matrix of 
%   simulated model responses (one row per trial, corresponding to rows of 
%   DMAT). DMAT is built by the algorithm out of rows of DESIGNMAT.
%
%   DESIGNMAT can be omitted or left empty, in which case FUN needs to 
%   accept a parameter vector PARAMS and an array of trial numbers T, and 
%   returns a matrix of simulated responses, where the i-th row contains 
%   the simulated response for trial T(i) (the indices in T may repeat).
%
%   NLOGL = IBSLIKE(FUN,PARAMS,RESPMAT,DESIGNMAT,OPTIONS) uses options in 
%   structure OPTIONS to replace default values. (To be explained...)
%
%   NLOGL = IBSLIKE(...,VARARGIN) additional arguments are passed to FUN.
%
%   [NLOGL,NLOGLVAR] = IBSLIKE(...) also returns an estimate NLOGVAR of the
%   variance of the log likelihood.
%
%   [NLOGL,NLOGLVAR,EXITFLAG] = IBSLIKE(...) returns an EXITFLAG that 
%   describes the exit condition. Possible values of EXITFLAG and the 
%   corresponding exit conditions are
%
%    2  IBS terminated after reaching the maximum runtime specified by the 
%       user (the estimate can be arbitrarily biased).
%    1  IBS terminated after reaching the negative log-likelihood
%       threshold specified by the user (the estimate is biased).
%    0  Correct run of IBS; the estimate is unbiased.
%
%   [NLOGL,NLOGLVAR,EXITFLAG,OUTPUT] = IBSLIKE(...) returns a structure 
%   OUTPUT with additional information about the sampling.
%
%   OPTIONS = IBSLIKE('defaults') returns a basic default OPTIONS structure.
%
%   EXITFLAG = IBSLIKE('test') runs some tests. Here EXITFLAG is 0 if 
%   everything works correctly.
%   
%   Test code on binomial sampling:
%      p = 0.7; Ntrials = 100;                  % Define binomial probability
%      fun = @(x,dmat) rand(size(dmat)) < x;    % Simulating function
%      rmat = fun(p,NaN(Ntrials,1));            % Generate responses
%      [nlogL,nlogLvar,pc,output] = ibslike(fun,p,rmat);
%      nlogL_true = -log(p)*sum(rmat == 1) - log(1-p)*sum(rmat == 0);
%      fprintf('Ground truth: %.4g, Estimate: %.4g ± %.4g.\n',nlogL_true,nlogL,sqrt(nlogLvar));
%
%   Reference: 
%   van Opheusden*, B., Acerbi*, L. & Ma, W. J. (2020), "Unbiased and 
%   efficient log-likelihood estimation with inverse binomial sampling". 
%   (* equal contribution), arXiv preprint https://arxiv.org/abs/2001.03985
%
%   See also @.

%--------------------------------------------------------------------------
% IBS: Inverse Binomial Sampling for unbiased log-likelihood estimation
% To be used under the terms of the MIT License 
% (https://opensource.org/licenses/MIT).
%
%   Authors (copyright): Luigi Acerbi and Bas van Opheusden, 2020
%   e-mail: luigi.acerbi@{gmail.com,nyu.edu}, basvanopheusden@nyu.edu
%   URL: http://luigiacerbi.com
%   Version: 0.91
%   Release date: May 06, 2020
%   Code repository: https://github.com/lacerbi/ibs
%--------------------------------------------------------------------------

if nargin < 4; designMat = []; end
if nargin < 5; options = []; end

t0 = tic;

% Default options
% defopts.Display       = 'off';        % Level of display on screen
defopts.Nreps           = 10;           % # independent log-likelihood estimates per trial
defopts.NegLogLikeThreshold = Inf;      % Stop sampling if estimated nLL is above threshold (incompatible with vectorized sampling)
defopts.Vectorized      = 'auto';       % Use vectorized sampling algorithm with acceleration
defopts.Acceleration    = 1.5;          % Acceleration factor for vectorized sampling
defopts.NsamplesPerCall = 0;            % # starting samples per trial per function call (0 = choose automatically)
defopts.MaxIter         = 1e5;          % Maximum number of iterations (per trial and estimate)
defopts.ReturnPositive  = false;        % If true, the first returned output is the *positive* log-likelihood
defopts.ReturnStd       = false;        % If true, the second returned output is the standard deviation of the estimate
defopts.MaxTime         = Inf;          % Maximum time for a IBS call (in seconds)

%% If called with no arguments or with 'defaults', return default options
if nargout <= 1 && (nargin == 0 || (nargin == 1 && ischar(fun) && strcmpi(fun,'defaults')))
    if nargin < 1
        fprintf('Basic default options returned (type "help ibslike" for help).\n');
    end
    nlogL = defopts;
    return;
end

%% If called with the first argument as 'test', run test
if ischar(fun) && strcmpi(fun,'test')
    if nargin < 2; options = []; else; options = params; end
    figure; 
    subplot(1,3,1);
    exitflag(1) = runtest1(options);
    subplot(1,3,2);
    exitflag(2) = runtest2(options);
    subplot(1,3,3);
    exitflag(3) = runtest3(options);
    nlogL = any(exitflag);
    return;
end

for f = fields(defopts)'
    if ~isfield(options,f{:}) || isempty(options.(f{:}))
        options.(f{:}) = defopts.(f{:});
    end
end

Ntrials = size(respMat,1);

% Add hard-coded options
options.MaxSamples = 1e4;                       % Maximum # of samples per function call
options.AccelerationThreshold = 0.1;            % Keep accelerating until threshold is passed (in s)
options.VectorizedThreshold  = 0.1;             % Max threshold for using vectorized algorithm (in s)
options.MaxMem = 1e6;                           % Maximum number of samples for vectorized implementation
options.MaxMem = max(min(Ntrials,1e4),10)*100;  % Maximum number of samples for vectorized implementation

% NSAMPLESPERCALL should be a scalar integer
if ~isnumeric(options.NsamplesPerCall) || ~isscalar(options.NsamplesPerCall)
    error('ibslike:NsamplesPerCall','OPTIONS.NsamplesPerCall should be a scalar integer.');
end

% ACCELERATION should be a scalar equal or greater than 1
if ~isnumeric(options.Acceleration) || ~isscalar(options.Acceleration) || ...
        options.Acceleration < 1
    error('ibslike:Acceleration','OPTIONS.Acceleration should be a scalar equal or greater than one.');
end

% NEGLOGLIKETHRESHOLD should be a scalar greater than 0 (or Inf)
if ~isnumeric(options.NegLogLikeThreshold) || ~isscalar(options.NegLogLikeThreshold) || ...
        options.NegLogLikeThreshold <= 0
    error('ibslike:NegLogLikeThreshold','OPTIONS.NegLogLikeThreshold should be a positive scalar (including Inf).');
end

% MAXTIME should be a positive scalar (including Inf)
if ~isnumeric(options.MaxTime) || ~isscalar(options.MaxTime) || ...
        options.MaxTime <= 0
    error('ibslike:MaxTime','OPTIONS.MaxTime should be a positive scalar (or Inf).');
end

Trials = (1:Ntrials)';
funcCount = 0;

simdata = []; elapsed_time = [];
if ischar(options.Vectorized) && options.Vectorized(1) == 'a'
    % First full simulation to determine computation time
    fun_clock = tic;
    if isempty(designMat)    % Pass only trial indices
        simdata = fun(params,Trials(:),varargin{:});
    else                    % Pass full design matrix per trial
        simdata = fun(params,designMat(Trials(:),:),varargin{:});   
    end
    elapsed_time = toc(fun_clock);
    vectorized_flag = elapsed_time < options.VectorizedThreshold;
    funcCount = 1;
else
    vectorized_flag = logical(options.Vectorized);
end

if vectorized_flag
    [nlogL,K,Nreps,Ns,fc,exitflag] = ...
        vectorized_ibs_sampling(fun,params,respMat,designMat,simdata,elapsed_time,t0,options,varargin{:});
else
    [nlogL,K,Nreps,Ns,fc,exitflag] = ...
        loop_ibs_sampling(fun,params,respMat,designMat,simdata,elapsed_time,t0,options,varargin{:});
end
funcCount = funcCount + fc;

% Variance of estimate per trial
if nargout > 1
    K_max = max(max(K(:),1));
    Ktab = -(psi(1,1:K_max)' - psi(1,1));    
    LLvar = Ktab(max(K,1));   
    nlogLvar = sum(LLvar,2)./Nreps.^2;
end

% OUTPUT structure with additional information
if nargout > 2
    output.funcCount = funcCount;
    output.NsamplesPerTrial = Ns/Ntrials;
    output.nlogL_trials = nlogL;
    output.nlogLvar_trials = nlogLvar;
end

% Return negative log-likelihood and variance summed over trials
nlogL = sum(nlogL);
if options.ReturnPositive; nlogL = -nlogL; end
if nargout > 1
    nlogLvar = sum(nlogLvar);
    if options.ReturnStd    % Return standard deviation instead of variance
        nlogLvar = sqrt(nlogLvar);
    end
end

end

%--------------------------------------------------------------------------
function [nlogL,K,Nreps,Ns,fc,exitflag] = vectorized_ibs_sampling(fun,params,respMat,designMat,simdata0,elapsed_time0,t0,options,varargin)

Ntrials = size(respMat,1);
Trials = (1:Ntrials)';
Ns = 0;
fc = 0;
exitflag = 0;

Psi_tab = [];   % Empty PSI table

% Empty matrix of K values (samples-to-hit) for each repeat for each trial
K_mat = zeros([max(options.Nreps),Ntrials]);

% Matrix of rep counts
K_place0 = repmat((1:size(K_mat,1))',[1,Ntrials]);

% Current rep being sampled for each trial
Ridx = ones(1,Ntrials);

% Current vector of "open" K values per trial (not reached a "hit" yet)
K_open = zeros(1,Ntrials);

targetHits = options.Nreps(:)'.*ones(1,Ntrials);
MaxIter = options.MaxIter*max(options.Nreps);

% Starting samples
if options.NsamplesPerCall == 0
    samples_level = options.Nreps;
else
    samples_level = options.NsamplesPerCall;
end

for iter = 1:MaxIter
    % Pick trials that need more hits, sample multiple times
    T = Trials(Ridx <= targetHits);
    if isfinite(options.MaxTime) && toc(t0) > options.MaxTime
        T = []; 
        exitflag = 2; 
    end
    if isempty(T); break; end
    
    Ttrials = numel(T);    % Number of trials under consideration
        
    % With accelerated sampling, might request multiple samples at once
    Nsamples = min(options.MaxSamples,max(1,round(samples_level)));
    MaxSamples = ceil(options.MaxMem / Ttrials);
    Nsamples = min(Nsamples, MaxSamples);
    Tmat = repmat(T,[1,Nsamples]);
    
    % Simulate trials
    if iter == 1 && Nsamples == 1 && ~isempty(simdata0)
        simdata = simdata0;
        elapsed_time = elapsed_time0;
    else
        fun_clock = tic;
        if isempty(designMat)    % Pass only trial indices
            simdata = fun(params,Tmat(:),varargin{:});
            fc = fc + 1;
        else                    % Pass full design matrix per trial
            simdata = fun(params,designMat(Tmat(:),:),varargin{:});   
            fc = fc + 1;
        end
        elapsed_time = toc(fun_clock);
    end
    Ns = Ns + Ttrials;
    
    % Accelerated sampling
    if options.Acceleration > 0 && elapsed_time < options.AccelerationThreshold
        samples_level = samples_level*options.Acceleration;
    end
    
    % Check new "hits"
    hits_temp = all(respMat(Tmat(:),:) == simdata,2);
    
    % Build matrix of new hits (sandwich with buffer of hits, then removed)
    hits_new = [ones(1,Ttrials);reshape(hits_temp,size(Tmat))';ones(1,Ttrials)];
            
    % Warning: from now on it's going to be incomprehensible 
    % (all vectorized for speed)
    
    % Extract matrix of Ks from matrix of hits for this iteration
    h = size(hits_new,1);
    list = find(hits_new(:) == 1)-1;
    row = floor(list/h)+1;
    col = mod(list,h)+1;
    delta = diff([col;1]);
    remidx = delta <= 0;
    delta(remidx) = [];
    row(remidx) = [];
    indexcol = find(diff([0;row]));
    col = 1 + (1:numel(row))' - indexcol(row);
    K_iter = zeros(size(T,1),max(col));
    K_iter(row + (col-1)*size(K_iter,1)) = delta;

    % This is the comprehensible version that we want to get to:
    %
    %   for iTrial = 1:Ntrials
    %       index = find(hits_new(iTrial,:),targetHits(iTrial));
    %       K = diff([0 index]);
    %       logL(iTrial) = sum(Ktab(K))/numel(index);
    %   end
    
    
    % Add still-open K to first column
    K_iter(:,1) = K_open(T)' + K_iter(:,1);
        
    % Find last K position for each trial
    [~,idx_last] = min([K_iter,zeros(Ttrials,1)],[],2);
    idx_last = idx_last - 1;
    ii = sub2ind(size(K_iter),(1:Ttrials)',idx_last);
    
    % Subtract one hit from last K (it was added)
    K_iter(ii) = K_iter(ii) - 1;
    K_open(T) = K_iter(ii)';
    
    % For each trial, ignore entries of K_iter past max # of reps
    idx_mat = bsxfun(@plus,Ridx(T)',repmat(0:size(K_iter,2)-1,[Ttrials,1]));
    K_iter(idx_mat > (options.Nreps)) = 0;
    
    % Find last K position for each trial again
    [~,idx_last2] = min([K_iter,zeros(Ttrials,1)],[],2);
    idx_last2 = idx_last2 - 1;
        
    % Add current K to full K matrix    
    K_iter_place = bsxfun(@ge,K_place0(:,1:Ttrials),Ridx(T)) & bsxfun(@le,K_place0(:,1:Ttrials),Ridx(T) + idx_last2'- 1);
    K_place = false(size(K_place0));
    K_place(:,T) = K_iter_place;
    Kt = K_iter';    
    K_mat(K_place) = Kt(Kt > 0);
    Ridx(T) = Ridx(T) + idx_last' - 1;
    
    % Compute log-likelihood only if requested for thresholding
    if isfinite(options.NegLogLikeThreshold)
        Rmin = min(Ridx(T));    % Find repeat still ongoing
        if Rmin > size(K_mat,1); continue; end
        [LL_temp,Psi_tab] = get_LL_from_K(Psi_tab,K_mat(Rmin,:));
        nLL_temp = -sum(LL_temp,2);
        if nLL_temp > options.NegLogLikeThreshold
            idx_move = Ridx == Rmin;
            Ridx(idx_move) = Rmin+1;
            K_open(idx_move) = 0;
            exitflag = 1;
        end
    end
end

if ~isempty(T)
    error('ibslike:ConvergenceFail', ...
        'Maximum number of iterations reached and algorithm did not converge. Check FUN and DATA.');
end
    
% Log likelihood estimate per trial and run lengths K for each repetition
Nreps = sum(K_mat > 0,1)';
[LL_mat,Psi_tab] = get_LL_from_K(Psi_tab,K_mat);
nlogL = sum(-LL_mat',2)./Nreps;
K = K_mat';

end



%--------------------------------------------------------------------------
function [nlogL,K,Nreps,Ns,fc,exitflag] = loop_ibs_sampling(fun,params,respMat,designMat,simdata0,elapsed_time0,t0,options,varargin)

Ntrials = size(respMat,1);
Trials = (1:Ntrials)';
MaxIter = options.MaxIter;
exitflag = 0;

K = zeros(Ntrials,options.Nreps);
Ns = 0;
fc = 0;
Psi_tab = [];

for iRep = 1:options.Nreps
    
    offset = 1;
    hits = zeros(Ntrials,1);
    if isfinite(options.MaxTime) && toc(t0) > options.MaxTime
        exitflag = 2;
        break;
    end
    
    for iter = 1:MaxIter
        % Pick trials that need more hits, sample multiple times
        T = Trials(hits < 1);
        if isempty(T); break; end        

        % Simulate trials
        if iter == 1 && iRep == 1 && ~isempty(simdata0)
            simdata = simdata0;
        elseif isempty(designMat)    % Pass only trial indices
            simdata = fun(params,T(:),varargin{:});
            fc = fc + 1;
        else                    % Pass full design matrix per trial
            simdata = fun(params,designMat(T(:),:),varargin{:});   
            fc = fc + 1;
        end
        Ns = Ns + numel(T); % Count samples
        hits_new = all(respMat(T(:),:) == simdata,2);    
        hits(T) = hits(T) + hits_new;
        
        K(T(hits_new),iRep) = offset;        
        offset = offset + 1;
        
        % Terminate if negative log likelihood is above a given threshold
        if isfinite(options.NegLogLikeThreshold)
            K(hits < 1,iRep) = offset;
            [LL_mat,Psi_tab] = get_LL_from_K(Psi_tab,K(:,iRep));
            nlogL_sum = -sum(LL_mat,1);            
            if nlogL_sum > options.NegLogLikeThreshold
                T = [];
                exitflag = 1;
                break;
            end
        end
        
        % Terminate if above maximum allowed runtime
        if isfinite(options.MaxTime) && toc(t0) > options.MaxTime
            T = [];
            exitflag = 2;
            break;
        end

    end    
end

if ~isempty(T)
    error('ibslike:ConvergenceFail', ...
        'Maximum number of iterations reached and algorithm did not converge. Check FUN and DATA.');
end
    
Nreps = sum(K > 0,2);
[LL_mat,Psi_tab] = get_LL_from_K(Psi_tab,K);
nlogL = sum(-LL_mat,2)./Nreps;

end


%--------------------------------------------------------------------------
function [LL_mat,Psi_tab] = get_LL_from_K(Psi_tab,K_mat)
%GET_LL_FROM_K Convert matrix of K values into log-likelihoods.

K_max = max(1,max(K_mat(:)));
if K_max > numel(Psi_tab)   % Fill digamma function table
    Psi_tab = [Psi_tab; (psi(1) - psi(numel(Psi_tab)+1:K_max)')];
end
LL_mat = Psi_tab(max(1,K_mat));

end

%--------------------------------------------------------------------------
function exitflag = runtest1(options)

Nreps = 1e3;
RMSE_tol = 2/sqrt(Nreps);

% Binomial probability model
p_model = exp(linspace(log(1e-3),log(1),10));
fun = @(x,dmat) rand(size(dmat)) < x;   % Simulating function
rmat = fun(1,NaN);

fprintf('\n');
fprintf('TEST 1: Using IBS to compute log(p) of Bernoulli distributions with %d repeats.\n',Nreps);
fprintf('We consider p = %s.\n',mat2str(p_model,3));

options.Nreps = Nreps;
options.NegLogLikeThreshold = Inf;

nlogL = zeros(1,numel(p_model));
nlogLvar = zeros(1,numel(p_model));
for iter = 1:numel(p_model)
    [nlogL(iter),nlogLvar(iter)] = ibslike(fun,p_model(iter),rmat,[],options);
end

% We expect the true value to be almost certainly (> 99.99%) in this range
LL_min = (-nlogL - 4*sqrt(nlogLvar));
LL_max = (-nlogL + 4*sqrt(nlogLvar));

exitflag = any(log(p_model) < LL_min) | any(log(p_model) > LL_max);

rmse = sqrt(mean((-nlogL - log(p_model)).^2));
fprintf('Average RMSE of log(p) estimates across p: %.4f.\n',rmse);

exitflag = exitflag | (rmse > RMSE_tol);

if exitflag
    fprintf('Test FAILED. Something might be wrong.\n');    
else
    fprintf('Test PASSED. IBS estimates are calibrated and close to ground truth.\n');    
end

% Plot figure
xx = log(p_model);
h(1) = plot(xx,xx,'k-','LineWidth',2); hold on;

yy = -nlogL;
xxerr = [xx, fliplr(xx)];
yyerr_down = yy - 1.96*sqrt(nlogLvar);
yyerr_up = yy + 1.96*sqrt(nlogLvar);
yyerr = [yyerr_down, fliplr(yyerr_up)];
fill(xxerr, yyerr,'b','FaceAlpha',0.5,'LineStyle','none'); hold on;
h(2) = plot(xx,yy,'b-','LineWidth',2); hold on;

box off;
set(gca,'TickDir','out');
set(gcf,'Color','w');
xlabel('True log(p)');
ylabel('Estimated log(p)')
%xlim([-5 5]);
hl = legend(h,'True log(p)','IBS estimate (95% CI)');
set(hl,'Location','NorthWest','Box','off');
title('IBS estimation test');

end


%--------------------------------------------------------------------------
function exitflag = runtest2(options)

% Binomial probability model
Ntrials = 100;                  
p_true = 0.9*rand() + 0.05;             % True probability
p_model = 0.9*rand() + 0.05;            % Model probability
fun = @(x,dmat) rand(size(dmat)) < x;   % Simulating function
Nexps = 2e3;

options.NegLogLikeThreshold = Inf;

fprintf('\n');
fprintf('TEST 2: Using IBS to compute the log-likelihood of a binomial distribution.\n');
fprintf('Parameters: p_true=%.2g, p_model=%.2g, %d trials per experiment.\n',p_true,p_model,Ntrials);
fprintf('The distribution of z-scores should approximate a standard normal distribution (mean 0, SD 1).\n');

zscores = zeros(1,Nexps);
for iter = 1:Nexps
    rmat = fun(p_true,NaN(Ntrials,1));            % Generate data
    [nlogL,nlogLvar] = ibslike(fun,p_model,rmat,[],options);
    nlogL_exact = -log(p_model)*sum(rmat == 1) - log(1-p_model)*sum(rmat == 0);
    zscores(iter) = (nlogL_exact - nlogL)/sqrt(nlogLvar);
end

edges = -4.75:0.5:4.75;
nz = histc(zscores,edges);
h(1) = bar(edges,nz,'histc');
hold on;
xx = linspace(-5,5,1e4);
h(2) = plot(xx,Nexps*exp(-xx.^2/2)/sqrt(2*pi)/2,'k-','LineWidth',2);

box off;
set(gca,'TickDir','out');
set(gcf,'Color','w');
xlabel('z-score');
ylabel('pdf')
xlim([-5 5]);
hl = legend(h,'z-scores histogram','expected pdf');
set(hl,'Location','NorthEast','Box','off');
title('Calibration test');

exitflag = abs(mean(zscores)) > 0.15 || abs(std(zscores) - 1) > 0.1;

fprintf('Distribution of z-scores (%d experiments). Mean: %.4g. Standard deviation: %.4g.\n',Nexps,mean(zscores),std(zscores));
if exitflag
    fprintf('Test FAILED. Something might be wrong.\n');    
else
    fprintf('Test PASSED. We verified that IBS is unbiased (~zero mean) and calibrated (SD ~1).\n');
end

end

%--------------------------------------------------------------------------
function exitflag = runtest3(options)

Nreps = 100;
RMSE_tol = 4/sqrt(Nreps);

% Binomial probability model
p_model = exp(linspace(log(1e-3),log(0.1),10));
fun = @(x,dmat) rand(size(dmat)) < x;   % Simulating function
rmat = fun(1,NaN);
thresh = -log(0.01);
p_target = max(p_model,exp(-thresh));

fprintf('\n');
fprintf('TEST 3: Log-likelihood thresholding at log(p) = %.3f.\n',-thresh);
fprintf('Using IBS to compute thresholded log(p) of Bernoulli distributions with %d repeats.\n',Nreps);
fprintf('We consider p = %s.\n',mat2str(p_model,3));

options.Nreps = Nreps;
options.NegLogLikeThreshold = thresh;
options.Acceleration = 1;

nlogL = zeros(1,numel(p_model));
nlogLvar = zeros(1,numel(p_model));
for iter = 1:numel(p_model)
    [nlogL(iter),nlogLvar(iter)] = ibslike(fun,p_model(iter),rmat,[],options);
end

% We expect the true value to be almost certainly (> 99.99%) in this range
LL_min = (-nlogL - 4*sqrt(nlogLvar));
LL_max = (-nlogL + 4*sqrt(nlogLvar));

% We expect the estimates to be (almost) correct away from the threshold
idx = log(p_model) > -thresh*0.75;
exitflag = any(log(p_model(idx)) < LL_min(idx)) | any(log(p_model(idx)) > LL_max(idx));

% We expect the estimates to be above the true value below the threshold
LL_thresh = (-nlogL - sqrt(nlogLvar));
idx_below = log(p_model) < -thresh;
exitflag = exitflag | any(log(p_model(idx_below)) > LL_thresh(idx_below));
% exitflag = any(log(p_target) < LL_min) | any(log(p_target) > LL_max);

rmse = sqrt(mean((-nlogL(idx) - log(p_target(idx))).^2));
fprintf('Average RMSE of log(p) estimates across p: %.4f.\n',rmse);

exitflag = exitflag | (rmse > RMSE_tol);

if exitflag
    fprintf('Test FAILED. Something might be wrong.\n');    
else
    fprintf('Test PASSED. IBS estimates are calibrated and close to (thresholded) ground truth.\n');    
end

% Plot figure
xx = log(p_model);
h(1) = plot(xx,xx,'k-','LineWidth',2); hold on;

yy = -nlogL;
xxerr = [xx, fliplr(xx)];
yyerr_down = yy - 1.96*sqrt(nlogLvar);
yyerr_up = yy + 1.96*sqrt(nlogLvar);
yyerr = [yyerr_down, fliplr(yyerr_up)];
fill(xxerr, yyerr,'b','FaceAlpha',0.5,'LineStyle','none'); hold on;

h(2) = plot(xx,yy,'b-','LineWidth',2); hold on;
h(3) = plot([xx(1),xx(end)],-thresh*[1 1],'k:','LineWidth',2);

box off;
set(gca,'TickDir','out');
set(gcf,'Color','w');
xlabel('True log(p)');
ylabel('Estimated log(p) with thresholding')
%xlim([-5 5]);
hl = legend(h,'True log(p)','IBS estimate (95% CI)','Threshold');
set(hl,'Location','NorthWest','Box','off');
title('Thresholded IBS test');

end


%   TODO:
%   - Fix help and documentation
%   - Optimal allocation of estimates?