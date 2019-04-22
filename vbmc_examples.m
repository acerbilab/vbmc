%VBMC_EXAMPLES Examples and tutorial for Variational Bayesian Monte Carlo
%
%  Example 1: Basic usage
%  Example 2: Non-bound constraints
%  Example 3: Extended usage and output diagnostics
%  Example 4: Multiple runs as validation
%
%  Note: after installation, run 
%    vbmc('test') 
%  to check that everything is working correctly.
%
%  For any question, check out the FAQ: 
%  https://github.com/lacerbi/vbmc/wiki
%
%  See also VBMC.

% Luigi Acerbi 2018

fprintf('Running some examples usage for Variational Bayesian Monte Carlo (VBMC).\n');
fprintf('Open ''vbmc_examples.m'' in the editor to read detailed comments along the tutorial.\n');


%% Example 1: Basic usage

close all;

fprintf('\n*** Example 1: Basic usage\n');
fprintf('  Simple usage of VBMC having as target log likelihood <a href="https://en.wikipedia.org/wiki/Rosenbrock_function">Rosenbrock''s banana function</a>.\n');
fprintf('  Press any key to continue.\n\n');
pause;

D = 2;                          % We consider a 2-D problem

% We set as toy log likelihood for our model a broad Rosenbrock's banana 
% function in 2D (see https://en.wikipedia.org/wiki/Rosenbrock_function).

llfun = @rosenbrock_test;

% LLFUN would be the function handle to the log likelihood of your model.
 
% We define now a prior over the parameters (for simplicity, independent
% Gaussian prior on each variable, but you could do whatever).

prior_mu = zeros(1,D);
prior_var = 3^2*ones(1,D);
lpriorfun = @(x) ...
    -0.5*sum((x-prior_mu).^2./prior_var,2) ...
    -0.5*log(prod(2*pi*prior_var));

% So our log joint (that is, unnormalized log posterior density), is:
fun = @(x) llfun(x) + lpriorfun(x);

% We assume an unconstrained domain for the model parameters, and finite 
% plausible bounds which should denote a region of high posterior 
% probability mass. Not knowing better, we use mean +/- 1 SD of the prior 
% (that is, the top ~68% prior credible interval) to set plausible bounds.

LB = -Inf(1,D);                            % Lower bounds
UB = Inf(1,D);                             % Upper bounds
PLB = prior_mu - sqrt(prior_var);          % Plausible lower bounds
PUB = prior_mu + sqrt(prior_var);          % Plausible upper bounds

% Analogously, you could set the plausible bounds using the quantiles:
% PLB = norminv(0.1587,prior_mu,sqrt(prior_var));
% PUB = norminv(0.8413,prior_mu,sqrt(prior_var));

% As a starting point, we use the mean of the prior:
x0 = prior_mu;

% Alternatively, we could have used a sample from inside the plausible box:
% x0 = PLB + rand(1,D).*(PUB - PLB);

% For now, we use default options for the inference:
options = vbmc('defaults');

% Run VBMC, which returns the variational posterior VP, the lower bound on
% the log model evidence ELBO, and its uncertainty ELBO_SD.
[vp,elbo,elbo_sd] = vbmc(fun,x0,LB,UB,PLB,PUB,options);

fprintf('  The true log model evidence is lnZ = -2.272.\n\n');
fprintf('  Note that:\n  1) The ELBO is a lower bound on the true log model evidence.');
fprintf('\n  2) The reported standard deviation of the ELBO is a measure of the uncertainty on the ELBO\n     as estimated via Bayesian quadrature (the approximation technique used by VBMC).\n');
fprintf('     It is NOT a measure of the difference between the ELBO and the true log model evidence,\n     which is generally unknown.\n');

% Note that VBMC does not aim for high numerical precision (e.g., beyond 
% the 1st or 2nd decimal place). In most realistic model-fitting problems,
% a higher resolution is not particularly useful.

fprintf('\n  We can now examine the obtained variational posterior.')
fprintf('  Press any key to continue.\n\n');
pause;

% First, let us generate a million samples from the variational posterior:
Xs = vbmc_rnd(vp,1e6);

% Easily compute statistics such as moments, credible intervals, etc.
post_mean = mean(Xs,1);                   % Posterior mean
post_cov = cov(Xs);                       % Posterior covariance matrix
% post_std = std(Xs,[],1);                % Posterior SD
% post_iqr = quantile(Xs,[0.25,0.75],1)   % Posterior interquartile ranges

% For reporting uncertainty on model parameter estimates, you could use
% posterior mean +/- SD, or the median and interquartile range (the latter
% is better for a posterior that deviates substantially from Gaussian).

fprintf('  The approximate posterior mean is: %s.\n  The approximate posterior covariance matrix is: %s.\n', ...
    mat2str(post_mean,3), mat2str(post_cov,3));
fprintf('  The plot shows the 1-D and 2-D approximate posterior marginal distributions.\n');

% We visualize the posterior marginals using the CORNERPLOT function:
cornerplot(Xs,{'x_1','x_2'});

fprintf('  Press any key to continue.\n\n');
pause;

%% Example 2: Bound parameters

close all;

fprintf('\n*** Example 2: Bounded parameters\n');
fprintf('  As in Example 1, but we assume parameters are constrained to be positive.\n');
fprintf('  We also display the evolution of the variational posterior in each iteration.\n');
fprintf('  (In the plots, black circles represent points in the training set,\n');
fprintf('  and red circles points added to the training set in this iteration.)\n');
fprintf('  Press any key to continue.\n\n');
pause;

llfun = @rosenbrock_test;                   % Log likelihood

D = 2;  % Still in 2-D

% Since parameters are positive, we impose an exponential prior.
prior_tau = 3*ones(1,D);            % Length scale of the exponential prior
lpriorfun = @(x) -sum(x./prior_tau,2) -log(prod(prior_tau));

fun = @(x) llfun(x) + lpriorfun(x);         % Log joint

% Bound settings require some discussion:
% 1) the specified bounds are NOT included in the domain, so in this case, 
%    since we want to be the parameters to be positive, we can set LB = 0, 
%    knowing that the parameter will always be greater than zero.
LB = zeros(1,D);                                % Lower bounds

% 2) Currently, VBMC does not support half-bounds, so we need to specify
%    a finite upper bound (cannot be Inf). We pick something very large
%    according to our prior. However, do not go crazy here by picking 
%    something impossibly large, otherwise VBMC will fail.
UB = 10*prior_tau;                              % Upper bounds

% 3) Plausible bounds need to be meaningfully different from hard bounds,
%    so here we cannot pick PLB = 0. We follow the same strategy as the first
%    example, by picking the top ~68% credible interval of the prior.
PLB = 0.1728*prior_tau;         % PLB = expinv(0.1587,prior_tau);
PUB = 1.8407*prior_tau;         % PUB = expinv(0.8413,prior_tau);

% Good practice is to initialize VBMC in a region of high posterior density.
% For this example, we cheat and start in the proximity of the mode, which
% is near the optimum of the likelihood (which we know).
x0 = ones(1,D);             % Optimum of the Rosenbrock function
options = vbmc('defaults');
options.Plot = true;        % Plot iterations

% Run VBMC
[vp,elbo,elbo_sd] = vbmc(fun,x0,LB,UB,PLB,PUB,options);

fprintf('  The true log model evidence is lnZ = -1.836.\n\n');

fprintf('  We will examine now the obtained variational posterior.')
fprintf('  Press any key to continue.\n\n');
pause;

close all;

Xs = vbmc_rnd(vp,1e6);  % Generate samples from the variational posterior

% We compute the pdf of the approximate posterior on a 2-D grid
plot_lb = [0 0];
plot_ub = quantile(Xs,0.999);
x1 = linspace(plot_lb(1),plot_ub(1),400);
x2 = linspace(plot_lb(2),plot_ub(2),400);
[xa,xb] = meshgrid(x1,x2);  % Build the grid
xx = [xa(:),xb(:)];         % Convert grids to a vertical array of 2-D points
yy = vbmc_pdf(vp,xx);       % Compute PDF values on specified points

% Plot approximate posterior pdf (works only in 1-D and 2-D)
surf(x1,x2,reshape(yy,[numel(x1),numel(x2)]),'EdgeColor','none');
xlabel('x_1');
ylabel('x_2');
zlabel('Approximate posterior pdf');
set(gca,'TickDir','out');
set(gcf,'Color','w');

% Compute and plot approximate posterior mean
hold on;
post_mean = mean(Xs,1);
h(1) = plot3(post_mean(1),post_mean(2),vbmc_pdf(vp,post_mean),'dk');

% Find and plot approximate posterior mode
post_mode = vbmc_mode(vp);
h(2) = plot3(post_mode(1),post_mode(2),vbmc_pdf(vp,post_mode),'xr');

hl = legend(h,'Approximate posterior mean','Approximate posterior mode');
set(hl,'Location','NorthEast','Box','off');

% We do not particularly recommend to use the posterior mode as parameter 
% estimate (also known as maximum-a-posterior or MAP estimate), because it 
% tends to be more brittle (especially in the VBMC approximation) than the 
% posterior mean.

fprintf('  Have a look at the 3-D visualization of the approximate posterior.');
fprintf('  Press any key to continue.\n\n');
pause;

%% Example 3: Extended usage and output diagnostics

close all;
clear data;

% Extended usage of VBMC and a look at some output arguments.

fprintf('\n*** Example 3: Extended usage and output\n');
fprintf('  Extended usage of VBMC with a diagnostic look at the output arguments.\n');
fprintf('  Press any key to continue.\n\n');
pause;

D = 4;      % We consider a 4-D problem

% Function handle for function with multiple input arguments (e.g., here
% we add a translation of the input; but more in general you could be 
% passing additional data to your objective function).
llfun = @(x,data) rosenbrock_test(bsxfun(@plus, x, data.mu));

% Gaussian prior, as per Example 1
prior_mu = zeros(1,D);
prior_var = 3*ones(1,D).^2;
lpriorfun = @(x) ...
    -0.5*sum((x-prior_mu).^2./prior_var,2) ...
    -0.5*log(prod(2*pi*prior_var));

% Define toy data
data.mu = ones(1,D);    % Translate Rosenbrock function

% Log joint distribution
fun = @(x,mu) llfun(x,data) + lpriorfun(x);

LB = -Inf;                                      % Lower bounds
UB = Inf;                                       % Upper bounds
PLB = prior_mu - sqrt(prior_var);               % Plausible lower bounds
PUB = prior_mu + sqrt(prior_var);               % Plausible upper bounds

% In a typical inference scenario, we recommend to start from a "good"
% point (e.g., near the mode). A good way is to run a preliminary quick
% optimization (a more extensive optimization would not harm).
x0 = PLB + (PUB-PLB).*rand(1,D);    % Random point inside plausible box

fprintf('  First, we run a preliminary optimization to start from a relatively high posterior density region.\n');
fprintf('  (You may want to use <a href="https://github.com/lacerbi/bads">Bayesian Adaptive Direct Search (BADS)</a> here.)\n');

% Here we only run it once but consider using multiple starting points.
opt_options = [];
opt_options.MaxFunEvals = 50*D;
opt_options.Display = 'final';
x0 = fminsearch(@(x) -fun(x),x0,opt_options);

% Also, instead of FMINSEARCH you should use an efficient method, like BADS:
% x0 = bads(@(x) -fun(x),x0,PLB,PUB,PLB,PUB,[],opt_options);
% You can download BADS from here: https://github.com/lacerbi/bads

% For demonstration purposes, we run VBMC for too short
options = vbmc('defaults');
options.MaxFunEvals = 10*D;

fprintf('  For this demonstration, we run VBMC with a restricted budget, insufficient to achieve convergence.\n\n');

% Run VBMC and get more output info
[vp,elbo,elbo_sd,exitflag,output] = vbmc(fun,x0,LB,UB,PLB,PUB,options);

fprintf('  VBMC is warning us that convergence is doubtful. We look at the output to check for diagnostics.\n');
fprintf('  Press any key to continue.\n');
pause;

exitflag
fprintf('  An EXITFLAG of 0 means that the algorithm has exhausted the budget of function evals.\n');

output
fprintf('  In the OUTPUT struct:\n  - the ''convergencestatus'' field says ''%s'' (probable lack of convergence);\n  - the reliabiltiy index ''rindex'' is %.2f (rindex needs to be less than one).\n\n', ...
    output.convergencestatus,output.rindex);

fprintf('  Our diagnostics tell that this run has not converged, suggesting to increase the budget.\n');
fprintf('  Note that convergence to a solution does not mean that it is a *good* solution.\n');
fprintf('  You should always check the returned variational posteriors (ideally with multiple runs of VBMC).\n\n');

fprintf('  We can now rerun VBMC for longer and with a more informed initialization.\n');
fprintf('  Press any key to continue.\n\n');
pause;

% Instead of starting from scratch, we use the output variational posterior 
% from before to obtain a better initialization. This way we do not need 
% to specify the starting point or the hard and plausible bounds, which are 
% automatically set based on VP.

options.MaxFunEvals = 50*(D+2);
if exitflag ~= 1        % Retry fit if the previous one did not converge
    [vp,elbo,elbo_sd,exitflag,output] = vbmc(fun,vp,[],[],[],[],options);
end

fprintf('  Thanks to a better initialization, this run converged quickly.\n');
fprintf('  Press any key to continue.\n\n');
pause;

% Note that the fractional overhead of VBMC reported in OUTPUT is astronomical.
% The reason is that the objective function we are using is analytical and 
% extremely fast, which is not what VBMC is designed for. 
% In a realistic scenario, the objective function will be moderately costly
% (e.g., more than ~0.5 s per function evaluation), and the fractional 
% overhead should of order 1 or less.

fprintf('  Finally, note that you can tell VBMC to automatically retry a run which did not converge.\n');
fprintf('  To do so, set the RetryMaxFunEvals options to a nonzero value (e.g., equal to MaxFunEvals).\n');
fprintf('  Check this section in the vbmc_exmples.m file for more details.\n');

% The following code snippet automatically reruns VBMC on the same problem
% with a better initialization if the first run does not converge:
%
% options = vbmc('defaults');
% options.RetryMaxFunEvals = options.MaxFunEvals;
% [vp,elbo,elbo_sd,exitflag,output] = vbmc(fun,x0,LB,UB,PLB,PUB,options);

fprintf('  Press any key to continue.\n\n');
pause;


%% Example 4: Multiple runs as validation

close all;

fprintf('\n*** Example 4: Multiple VBMC runs as validation\n');
fprintf('  Practical example with multiple VBMC runs as sanity check.');
fprintf('  Press any key to continue.\n\n');
pause;

D = 2;                          % Back to 2-D (for speed of demonstration)
llfun = @rosenbrock_test;       % Log likelihood function
prior_mu = zeros(1,D);          % Define log prior
prior_var = 3.^2*ones(1,D);
lpriorfun = @(x) -0.5*sum((x-prior_mu).^2./prior_var,2) -0.5*log(prod(2*pi*prior_var));
fun = @(x) llfun(x) + lpriorfun(x);     % Target log joint

LB = -Inf(1,D);                                 % Lower bounds
UB = Inf(1,D);                                  % Upper bounds
PLB = prior_mu - sqrt(prior_var);               % Plausible lower bounds
PUB = prior_mu + sqrt(prior_var);               % Plausible upper bounds

options = vbmc('defaults');     % Default VBMC options

opt_options = [];               % Preliminary optimization options
opt_options.MaxFunEvals = 50*D;
opt_options.Display = 'final';

Nruns = 3;      % Perform multiple runs (we suggest at least 3-4)

fprintf('  For validation, we recommend to run VBMC multiple times (3-4) with different initializations.\n');

vp = []; elbo = []; elbo_sd = []; exitflag = [];
for iRun = 1:Nruns
    fprintf('  VBMC run #%d/%d...\n', iRun, Nruns);
    
    if iRun == 1
        % First run, start from prior mean
        x0 = prior_mu;
    else
        % Other starting points, randomize
        x0 = PLB + (PUB-PLB).*rand(1,D);
    end
    
    x0 = fminsearch(@(x) -fun(x),x0,opt_options);
    % x0 = bads(@(x) -fun(x),x0,LB,UB,PLB,PUB,[],opt_options);
        
    [vp{iRun},elbo(iRun),elbo_sd(iRun),exitflag(iRun)] = vbmc(fun,x0,LB,UB,PLB,PUB,options);
end

fprintf('  We now perform a number of diagnostic checks on the variational solutions from different runs.\n');
fprintf('  Press any key to continue.\n\n');
pause;

fprintf('  We check that ELBOs from different runs are close-by (e.g., differences << 1).\n');
elbo

fprintf('  Then, we check that the variational posteriors from distinct runs are similar.\n');
fprintf('  As a metric of similarity, we compute the Kullback-Leibler (KL) divergence between each pair.\n');

% Compute KL-divergence across all pairs of solutions
kl_mat = zeros(Nruns,Nruns);
for iRun = 1:Nruns
    for jRun = iRun+1:Nruns        
        kl = vbmc_kldiv(vp{iRun},vp{jRun});
        kl_mat(iRun,jRun) = kl(1);
        kl_mat(jRun,iRun) = kl(2);        
    end
end

fprintf('  Note that the KL divergence is asymmetric, so we have an asymmetric matrix.\n');
kl_mat
fprintf('  Ideally, we want all KL divergence matrix entries to be << 1.\n');
fprintf('  For a qualitative validation, we recommend to also visually inspect the variational posteriors.\n');

fprintf('\n  We can also check that convergence was achieved in all runs (we want EXITFLAG = 1).\n');
exitflag
fprintf('  Finally, we can pick the variational solution with highest ELCBO (lower confidence bound on the ELBO).\n');

beta_lcb = 3;       % Standard confidence parameter 
% beta_lcb = 5;     % This is more conservative
elcbo = elbo - beta_lcb*elbo_sd
[~,idx_best] = max(elcbo);
idx_best

fprintf('  Press any key to continue.\n\n');
pause;

fprintf('  The function vbmc_diagnostics performs a battery of similar diagnostic checks\n  on a set of solutions returns by VBMC.\n');
fprintf('  We run now vbmc_diagnostics on our previously obtained variational posteriors:\n\n');

[exitflag,best,idx_best,stats] = vbmc_diagnostics(vp);

fprintf('  Press any key to continue.\n\n');
pause;

fprintf('  In addition to the displayed information, vbmc_diagnostics returns an EXITFLAG representing\n');
fprintf('  the outcome of the tests, a struct BEST with the "best" solution, its index IDX_BEST in the\n');
fprintf('  solution array, and a struct of summary statistics STATS.\n');

exitflag
best
idx_best
stats

fprintf('\n  Press any key to continue.\n\n');
pause;

fprintf('  This is all for this tutorial.\n');
fprintf('  You can read more detailed comments by opening the file ''vbmc_examples.m'' in the editor.\n\n');
fprintf('  Type ''help vbmc'' for additional documentation on VBMC, or consult the <a href="https://github.com/lacerbi/vbmc">Github page</a> or <a href="https://github.com/lacerbi/vbmc/wiki">online FAQ</a>.\n\n');

