function [samples,fvals,exitflag,output] = slicesamplebnd(logf,x0,N,widths,LB,UB,options,varargin)
%SLICESAMPLEBND Slice sampling method with bounds.
%
%   SAMPLES = SLICESAMPLEBND(LOGF,X0,N) draws N random samples from a 
%   target distribution with the log probability density function LOGF 
%   using the coordinate-wise slice sampling method. X0 is a row vector or 
%   scalar containing the initial value of the random sample sequences. 
%   X0 must be within the domain of the target distribution. N is the number 
%   of samples to be returned. LOGF is a function handle created using @. 
%   LOGF takes one argument as input that has the same type and size as X0 
%   and returns the target log density function (minus a constant; that is,
%   the normalization constant of the pdf need not be known). SAMPLES is a 
%   matrix each row of which corresponds to a sampled point in the sequence.
%
%   SAMPLES = SLICESAMPLEBND(LOGF,X0,N,WIDTHS) uses WIDTHS as a scalar or 
%   vector of typical widths. If it is a scalar, all dimensions are assumed 
%   to have the same typical widths. If it is a vector, each element of the 
%   vector is the typical width of the marginal target distribution in that 
%   dimension. The default value of W(i) is (UB(i)-LB(i))/2 if the i-th 
%   bounds are finite, or 10 otherwise. By default SLICESAMPLEBND uses an 
%   adaptive widths method during the burn-in period, so the choice of 
%   typical widths is not crucial.
%
%   SAMPLES = SLICESAMPLEBND(LOGF,X0,N,WIDTHS,LB,UB) defines a set of lower 
%   and upper bounds on the domain of the target density function, which is
%   assumed to be zero outside the range LB <= X <= UB. Use empty matrices 
%   for LB and UB if no bounds exist. Set LB(i) = -Inf if X(i) is unbounded 
%   below; set UB(i) = Inf if X(i) is unbounded above. If LB(i) == UB(i),
%   the variable is assumed to be fixed on that dimension.
%
%   SAMPLES = SLICESAMPLEBND(LOGF,X0,N,WIDTHS,LB,UB,OPTIONS) samples with 
%   the default sampling parameters replaced by values in the structure 
%   OPTIONS. SLICESAMPLEBND uses these options:
%
%     OPTIONS.Thin generates random samples with Thin-1 out of Thin values 
%     omitted in the generated sequence (after burn-in). Thin is a positive
%     integer. The default value of Thin is 1.
%
%     OPTIONS.Burnin omits the first Burnin points before starting recording 
%     points for the generated sequence. Burnin is a non-negative integer. 
%     The default value of Burnin is round(N/3) (that is, one third of the 
%     number of recorded samples).
%    
%     OPTIONS.StepOut if set to true, performs the stepping-out action when 
%     the current window does not bracket the probability density; see 
%     Neal (2003) for details. StepOut is a boolean value. The default 
%     value of StepOut is false. 
%
%     OPTIONS.Display defines the level of display. Accepted values for
%     Display are 'iter', 'notify', 'final', and 'off' for no display. The 
%     default value of Display is 'off'. 
%    
%     OPTIONS.LogPrior allows the user to specify a prior over X.
%     OPTIONS.LogPrior is a function handle created using @. LogPrior takes 
%     one argument as input that has the same type and size as X0 and 
%     returns the log prior density function at X. The generated samples
%     will be then drawn from the log density LOGF + OPTIONS.LogPrior.
%    
%     OPTIONS.Adaptive specifies whether to adapt the widths WIDTHS at the
%     end of the burn-in period based on the samples obtained so far.
%     Adaptive is a boolean value. By default OPTIONS.Adaptive is set to 
%     true. Set it to false to disable the automatic adaptation of typical
%     widths (for example, if you already have good estimates).
%    
%     OPTIONS.Diagnostics specifies whether convergence diagnostics are
%     performed at the end of the run. Diagnostics is a boolean value. Set 
%     OPTIONS.Diagnostics to true to run the diagnostics, false to skip it. 
%     The default for OPTIONS.Diagnostics is true. The diagnostics tests 
%     use the PSRF function by Simo Sarkka and Aki Vehtari, which implements 
%     diagnostics from Gelman et al. (2013).
%
%   SAMPLES = SLICESAMPLEBND(...,VARARGIN) passes additional arguments
%   VARARGIN to LOGF.
%
%   [SAMPLES,FVALS] = SLICESAMPLEBND(...) returns the sequence of values 
%   FVALS of the target log pdf at the sampled points. If a prior is
%   specified in OPTIONS.LogPrior, FVALS does NOT include the contribution
%   of the prior.
%
%   [SAMPLES,FVALS,EXITFLAG] = SLICESAMPLEBND(...) returns an EXITFLAG that
%   describes the exit condition of SLICESAMPLEBND. Possible values of 
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
%    -2 Detected probable lack of convergence of the sampling procedure.
%    -3 Detected lack of convergence of the sampling procedure.
%
%   [SAMPLES,FVALS,EXITFLAG,OUTPUT] = SLICESAMPLEBND(...) returns a structure
%   OUTPUT with the number of evaluations of LOGF in OUTPUT.funcCount, the 
%   value WIDTHS used during sampling in OUTPUT.widths (they can differ from 
%   the initial WIDTHS due to width adaptation during the burn-in stage), 
%   and the sequence of the values of the log prior at the sampled points 
%   in OUTPUT.logpriors (only if OPTIONS.LogPrior is nonempty). OUTPUT also
%   contains results of the convergence diagnostics, if available, such as 
%   the Potential Scale Reduction Factor in OUTPUT.R, the number of 
%   effective samples in OUTPUT.Neff and the estimated autocorrelation time
%   in OUTPUT.tau.
%
%   LOGF can return either a scalar (the value of the log probability 
%   density at X) or a row vector (the value of the log probability density
%   at X for each data point; each column corresponds to a different data
%   point). In the latter case, the total log pdf is obtained by summing
%   the log pdf per each individual data point. If LOGPDF returns a vector,
%   FVALS returned by SLICESAMPLEBND is a matrix (each row corresponds to a 
%   sampled point, each column to a different data point).
%   Knowing the log pdf of the sampled points per each data point can be 
%   useful to compute estimates of predictive error such as the widely 
%   applicable information criterion (WAIC); see Watanabe (2010).

% Author:   Luigi Acerbi
% Email:    luigi.acerbi@{nyu.edu,gmail.com}
% Date:     Feb/25/2016
%
% Inspired by a MATLAB implementation of slice sampling by Iain Murray.
% See pseudo-code in MacKay (2003).
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

%% Default parameters and options

% By default unbounded sampling
if nargin < 4; widths = []; end
if nargin < 5 || isempty(LB); LB = -Inf; end
if nargin < 6 || isempty(UB); UB = Inf; end
if nargin < 7; options = []; end

% Default options
defopts.Thin        = 1;            % Thinning
defopts.Burnin      = round(N/3);   % Burn-in
defopts.StepOut     = false;        % Step-out
defopts.Display     = 'notify';     % Display
defopts.Adaptive    = true;         % Adaptive widths
defopts.LogPrior    = [];           % Log prior over X
defopts.Diagnostics = true;         % Convergence diagnostics at the end
defopts.MetropolisPdf = [];         % Metropolis proposal probability density function
defopts.MetropolisRnd = [];         % Random draw from Metropolis proposal density

for f = fields(defopts)'
    if ~isfield(options,f{:}) || isempty(options.(f{:}))
        options.(f{:}) = defopts.(f{:});
    end
end

%% Startup and initial checks
D = numel(x0);
if numel(LB) == 1; LB = repmat(LB, [1,D]); end
if numel(UB) == 1; UB = repmat(UB, [1,D]); end
if size(LB,1) > 1; LB = LB'; end
if size(UB,1) > 1; UB = UB'; end
if numel(widths) == 1; widths = repmat(widths, [1,D]); end
LB_out = LB - eps(LB);
UB_out = UB + eps(UB);
basewidths = widths;    % User-supplied widths
if isempty(options.LogPrior); doprior = 0; else doprior = 1; end

if options.Burnin == 0 && isempty(widths) && options.Adaptive
    warning('WIDTHS not specified and adaptation is ON (OPTIONS.Adaptive == 1), but OPTIONS.Burnin is set to 0. SLICESAMPLEBND will attempt to use default values for WIDTHS.');
end

if isempty(widths); widths = (UB - LB)/2; end
widths(isinf(widths)) = 10;

funccount = 0;
[y,fval,logprior] = feval(@logpdfbound, x0, varargin{:});   % First evaluation at X0
xx = x0;
samples = zeros(N, D);
if ~isempty(options.LogPrior) && nargout > 3; logpriors = zeros(N, 1); else logpriors = []; end
if nargout > 1; fvals = zeros(N, numel(fval)); end
thin = floor(options.Thin);
burn = floor(options.Burnin);
log_Px = y;
widths(LB == UB) = 1;   % WIDTHS is irrelevant when LB == UB, set to 1

metropolis_flag = ~isempty(options.MetropolisPdf) || ~isempty(options.MetropolisRnd);

% Sanity checks
assert(size(x0,1) == 1 && size(x0,1) == 1, ...
    'The initial point X0 needs to be a scalar or row vector.');
assert(size(LB,1) == 1 && size(UB,1) == 1 && numel(LB) == D && numel(UB) == D, ...
    'LB and UB need to be empty matrices, scalars or row vectors of the same size as X0.');
assert(all(UB >= LB), ...
    'All upper bounds UB need to be equal or greater than lower bounds LB.');
assert(all(widths > 0 & isfinite(widths)) && isreal(widths), ...
    'The vector WIDTHS need to be all positive real numbers.');
assert(all(x0 >= LB) & all(x0 <= UB), ...
    'The initial starting point X0 is outside the bounds.');
assert(all(isfinite(y)) && isreal(y), ...
    'The initial starting point X0 needs to evaluate to a real number (not Inf or NaN).');
assert(thin > 0 && isscalar(thin), ...
    'The thinning factor OPTIONS.Thin needs to be a positive integer.');
assert(burn >= 0 && isscalar(burn), ...
    'The burn-in samples OPTIONS.Burnin needs to be a non-negative integer.');
assert(~metropolis_flag || ...
    (isa(options.MetropolisPdf,'function_handle') && isa(options.MetropolisRnd,'function_handle')), ...
    'Both OPTIONS.MetropolisPdf and OPTIONS.MetropolisRnd, if specified, need to be function handles.');

effN = N + (N-1)*(thin-1); % Effective samples

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

if trace > 1
    fprintf(' Iteration     f-count       log p(x)                   Action\n');
    displayFormat = ' %7.0f     %8.0f    %12.6g    %26s\n';
end

xx_sum = zeros(1,D);
xx_sqsum = zeros(1,D);

% Main loop
for ii = 1:(effN+burn)

    if trace > 1 && ii == burn+1
        action = 'start recording';
        fprintf(displayFormat,ii-burn,funccount,log_Px,action);
    end
    
    %% Metropolis step (optional)
    
    if metropolis_flag
        [xx,log_Px,fval,logprior] = metropolis_step(xx,varargin{:});
    end
    
    %% Slice-sampling step

    % Random-permutation axes sweep    
    dvec = randperm(D);
    for idd = 1:D
        dd = dvec(idd);
        
        if LB(dd) == UB(dd); continue; end      % Fixed dimension, skip
    
        log_uprime = log(rand) + log_Px;
        
        x_l = xx;
        x_r = xx;
        xprime = xx;

        % Create a horizontal interval (x_l, x_r) enclosing xx
        rr = rand;
        x_l(dd) = xx(dd) - rr*widths(dd);
        x_r(dd) = xx(dd) + (1-rr)*widths(dd);
        
        % Adjust interval to outside bounds for bounded problems
        if isfinite(LB(dd)) || isfinite(UB(dd))        
            if x_l(dd) < LB_out(dd)
                delta = LB_out(dd) - x_l(dd);
                x_l(dd) = x_l(dd) + delta;
                x_r(dd) = x_r(dd) + delta;
            end
            if x_r(dd) > UB_out(dd)
                delta = x_r(dd) - UB_out(dd);
                x_l(dd) = x_l(dd) - delta;
                x_r(dd) = x_r(dd) - delta;
            end
            x_l(dd) = max(x_l(dd),LB_out(dd));
            x_r(dd) = min(x_r(dd),UB_out(dd));
        end
        
        % Step-out procedure
        if options.StepOut
            steps = 0;
            stepsize = widths(dd);
            while (feval(@logpdfbound, x_l, varargin{:}) > log_uprime)
                x_l(dd) = x_l(dd) - stepsize;
                steps = steps + 1;
            end
            while (feval(@logpdfbound, x_r, varargin{:}) > log_uprime)
                x_r(dd) = x_r(dd) + stepsize;
                steps = steps + 1;
            end
            if trace > 1 && steps >= 10
                action = ['step-out dim ' num2str(dd) ' (' num2str(steps) ' steps)'];
                fprintf(displayFormat,ii-burn,funccount,log_Px,action);
            end            
        end

        % Shrink procedure (inner loop)
        % Propose xprimes and shrink interval until good one found
        shrink = 0;
        while 1
            shrink = shrink + 1;
            
            xprime(dd) = rand()*(x_r(dd) - x_l(dd)) + x_l(dd);
            [log_Px,fval,logprior] = feval(@logpdfbound, xprime, varargin{:});
                        
            if log_Px > log_uprime
                break % this is the only way to leave the while loop
            else
                % Shrink in
                if xprime(dd) > xx(dd)
                    x_r(dd) = xprime(dd);
                elseif xprime(dd) < xx(dd)
                    x_l(dd) = xprime(dd);
                else
                    %errorstr = ['Shrunk to current position and proposal still not acceptable. ' ...
                    %    'Current position: ' num2str(xx,' %g') '. ' ...
                    %    'Log f: (new value) ' num2str(log_Px), ', (target value) ' num2str(log_uprime) '.'];
                    %error(errorstr);
                    break;
                end
            end
        end
        
        % Width adaptation (only during burn-in, might break detailed balance)
        if ii <= burn && options.Adaptive
            delta = UB(dd) - LB(dd);
            if shrink > 3
                if isfinite(delta)
                    widths(dd) = max(widths(dd)/1.1,eps(delta));
                else
                    widths(dd) = max(widths(dd)/1.1,eps);                    
                end
            elseif shrink < 2
                widths(dd) = min(widths(dd)*1.2, delta);
            end
        end
        
        if trace > 1 && shrink >= 10
            action = ['shrink dim ' num2str(dd) ' (' num2str(shrink) ' steps)'];
            fprintf(displayFormat,ii-burn,funccount,log_Px,action);
        end
                
        xx(dd) = xprime(dd);
%        shrink
    end
    
    %% Metropolis step (optional)    

    if metropolis_flag
        [xx,log_Px,fval,logprior] = metropolis_step(xx,varargin{:});
    end
    
    
    %% Record samples and miscellaneous bookkeeping    
    
    % Record samples?
    record = ii > burn && mod(ii - burn - 1, thin) == 0;
    if record
        ismpl = 1 + (ii - burn - 1)/thin;
        samples(ismpl,:) = xx;
        if nargout > 1; fvals(ismpl,:) = fval(:); end
        if nargout > 3 && doprior; logpriors(ismpl) = logprior; end
    end
    
    % Store summary statistics starting half-way into burn-in
    if ii <= burn && ii > burn/2
        xx_sum = xx_sum + xx;
        xx_sqsum = xx_sqsum + xx.^2;
        
        % End of burn-in, update WIDTHS if using adaptive method
        if ii == burn && options.Adaptive
            burnstored = floor(burn/2);
            newwidths = 5*sqrt(xx_sqsum/burnstored - (xx_sum/burnstored).^2);
            newwidths = min(newwidths, UB_out - LB_out);
            if ~isreal(newwidths); newwidths = widths; end
            if isempty(basewidths)
                widths = newwidths;
            else
                % Max between new widths and geometric mean with user-supplied 
                % widths (i.e. bias towards keeping larger widths)
                widths = max(newwidths, sqrt(newwidths.*basewidths));
            end
        end
    end
        
    if trace > 2
        if ii <= burn; action = 'burn';
        elseif ~record; action = 'thin';
        else action = 'record';
        end
        fprintf(displayFormat,ii-burn,funccount,log_Px,action);
    end    
    
end

if trace > 0
    if thin > 1
        thinmsg = ['\n   and keeping 1 sample every ' num2str(thin) ', '];
    else
        thinmsg = '\n   ';
    end    
    fprintf(['\nSampling terminated:\n * %d samples obtained after a burn-in period of %d samples' thinmsg 'for a total of %d function evaluations.'], ...
        N, burn, funccount);
end

% Diagnostics
if options.Diagnostics && (nargout > 2 || trace > 0)
    [exitflag,R,Neff,tau] = diagnose(samples,trace,options);
    diagstr = [];
    if exitflag == -2 || exitflag == -3
        diagstr = [diagstr '\n * Try sampling for longer, by increasing N or OPTIONS.Thin.'];
    elseif exitflag == -1
        diagstr = [diagstr '\n * Try increasing OPTIONS.Thin to obtain more uncorrelated samples.'];
    elseif exitflag == 1
        diagstr = '\n * No violations of convergence have been detected (this does NOT guarantee convergence).';
    end
    if trace > 0 && ~isempty(diagstr)
        fprintf(diagstr);
    end
else
    exitflag = 0;
end

if trace > 0
    fprintf('\n\n');
end

if nargout > 3
    output.widths = widths;
    output.logpriors = logpriors;
    output.funccount = funccount;
    if options.Diagnostics
        output.R = R;
        output.Neff = Neff;
        output.tau = tau;
    end
end


%--------------------------------------------------------------------------
function [xx_up,y,fval_up,logprior_up] = metropolis_step(x,varargin)
%METROPOLIS_STEP Metropolis step.

    % Generate and evaluate Metropolis proposal
    xx_new = options.MetropolisRnd();        
    [log_Px_new,fval_new,logprior_new] = feval(@logpdfbound,xx_new,varargin{:});

    % Acceptance rate
    a = exp(log_Px_new - log_Px) * (options.MetropolisPdf(x) / options.MetropolisPdf(xx_new));

    % Accept proposal?
    if rand() < a
        xx_up = xx_new;        y = log_Px_new;
        fval_up = fval_new;    logprior_up = logprior_new;
    else
        xx_up = x;              y = log_Px;
        fval_up = fval;         logprior_up = logprior;
    end

end

%--------------------------------------------------------------------------
function [y,fval,logprior] = logpdfbound(x,varargin)
%LOGPDFBOUND Evaluate log pdf with bounds and prior.

    fval = [];
    logprior = [];

    if any(x < LB | x > UB)
        y = -Inf;
    else
        
        if doprior
            logprior = feval(options.LogPrior, x);
            if isnan(logprior)
                y = -Inf;
                warning('Prior density function returned NaN. Trying to continue.');
                return;
            elseif ~isfinite(logprior)
                y = -Inf;
                return;
            end
        else
            logprior = 0;
        end
        
        fval = logf(x,varargin{:});
        funccount = funccount + 1;
        
        if any(isnan(fval))
            y = -Inf;
            % warning('Target density function returned NaN. Trying to continue.');
        else
            y = sum(fval) + logprior;        
        end
    end
end

end

function [exitflag,R,Neff,tau] = diagnose(samples,trace,options)
%DIAGNOSE Perform quick-and-dirty diagnosis of convergence

    N = size(samples,1);
    exitflag = 0;

    try        
        warning_orig = warning;
        warning('off','all');
        [R,Neff,~,~,~,tau] = psrf(samples(1:floor(N/2),:), samples(floor(N/2)+(1:floor(N/2)),:));
        warning(warning_orig);
        
        diagstr = [];
        if any(R > 1.5)
            diagstr = ['\n * Detected lack of convergence! (max R = ' num2str(max(R),'%.2f') ' >> 1, mean R = ' num2str(mean(R),'%.2f') ').'];
            exitflag = -3;
        elseif any(R > 1.1)
            diagstr = ['\n * Detected probable lack of convergence (max R = ' num2str(max(R),'%.2f') ' > 1, mean R = ' num2str(mean(R),'%.2f') ').'];
            exitflag = -2;
        end
        if any(Neff < N/10)
            diagstr = [diagstr '\n * Low number of effective samples (min Neff = ' num2str(min(Neff), '%.1f') ...
                ', mean Neff = ' num2str(mean(Neff),'%.1f') ', requested N = ' num2str(N,'%d') ').'];
            if exitflag == 0; exitflag = -1; end
        end
        if isempty(diagstr) && exitflag == 0
            exitflag = 1;
        end
        
        if trace > 0 && ~isempty(diagstr)
            fprintf(diagstr);
        end
        
    catch ME
        warning(warning_orig);
        warning(['Error while computing convergence diagnostics with PSRF: ', ME.message]);
        R = NaN;
        Neff = NaN;
        tau = NaN;
    end

end
