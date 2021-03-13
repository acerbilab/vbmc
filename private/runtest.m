function failed = runtest(options)
%RUNTEST Test Variational Bayesian Monte Carlo (VBMC).
%  RUNTEST executes a few runs of the VBMC inference algorithm to
%  check that it is installed correctly, and returns the number of failed
%  tests.
% 
%  See also VBMC, VBMC_EXAMPLES.

tolerr = [0.5 0.5];     % Error tolerance
RunTime = 240;          % Ballpark runtime s on my laptop (fluctuates!)
specs = 'i7-9750H CPU @ 2.60GHz, 32 GB RAM';

if nargin < 1 || isempty(options); options = vbmc('defaults'); end

t = tic;

id = 1;
nvars = 6;
txt{id} = [num2str(id) '. Test with multivariate normal distribution (D = ' num2str(nvars) ', unconstrained)'];
fprintf('%s.\n', txt{id});
x0 = -ones(1,nvars);                   % Initial point
PLB = -2*nvars; PUB = 2*nvars;
LB = -Inf; UB = Inf;
lnZ = 0; mubar = zeros(1,nvars);
fun = @(x) sum(-0.5*(x./(1:numel(x))).^2) - sum(log(1:numel(x))) - 0.5*numel(x)*log(2*pi);
[exitflag(id),err(id,:)] = testblock(fun,x0,LB,UB,PLB,PUB,lnZ,mubar,options);

id = 2;
nvars = 2;
txt{id} = [num2str(id) '. Test with multivariate half-normal distribution (D = ' num2str(nvars) ', constrained)'];
fprintf('%s.\n', txt{id});
x0 = -ones(1,nvars);                   % Initial point
LB = -nvars*10; UB = 0;
PLB = -6; PUB = -0.05;
lnZ = -nvars*log(2); mubar = -2/sqrt(2*pi)*(1:nvars);
fun = @(x) sum(-0.5*(x./(1:numel(x))).^2) - sum(log(1:numel(x))) - 0.5*numel(x)*log(2*pi);
[exitflag(id),err(id,:)] = testblock(fun,x0,LB,UB,PLB,PUB,lnZ,mubar,options);

id = 3;
nvars = 3;
txt{id} = [num2str(id) '. Test with correlated multivariate normal distribution (D = ' num2str(nvars) ', unconstrained)'];
fprintf('%s.\n', txt{id});
x0 = 0.5*ones(1,nvars);                   % Initial point
LB = -Inf(1,nvars); UB = Inf(1,nvars);
PLB = -1; PUB = 1;
lnZ = 0; mubar = linspace(-0.5,0.5,nvars);
[exitflag(id),err(id,:)] = testblock(@cigar,x0,LB,UB,PLB,PUB,lnZ,mubar,options);

id = 4;
nvars = 3;
txt{id} = [num2str(id) '. Test with correlated multivariate normal distribution (D = ' num2str(nvars) ', constrained)'];
fprintf('%s.\n', txt{id});
x0 = 0.5*ones(1,nvars);                   % Initial point
LB = -4*ones(1,nvars); UB = 4*ones(1,nvars);
PLB = -1; PUB = 1;
lnZ = 0; mubar = linspace(-0.5,0.5,nvars);
[exitflag(id),err(id,:)] = testblock(@cigar,x0,LB,UB,PLB,PUB,lnZ,mubar,options);

id = 5;
nvars = 2;
txt{id} = [num2str(id) '. Test with noisy multivariate half-normal distribution (D = ' num2str(nvars) ', constrained)'];
fprintf('%s.\n', txt{id});
x0 = -ones(1,nvars);                   % Initial point
LB = -nvars*10; UB = 0;
PLB = -6; PUB = -0.05;
lnZ = -nvars*log(2); mubar = -2/sqrt(2*pi)*(1:nvars);
[exitflag(id),err(id,:)] = testblock(@noisefun,x0,LB,UB,PLB,PUB,lnZ,mubar,options,1);

id = 6;
nvars = 1;
txt{id} = [num2str(id) '. Test with uniform distribution (D = ' num2str(nvars) ', constrained)'];
fprintf('%s.\n', txt{id});
x0 = 0.5*ones(1,nvars);                   % Initial point
LB = zeros(1,nvars); UB = ones(1,nvars);
PLB = 0.05; PUB = 0.95;
lnZ = 0; mubar = 0.5*ones(1,nvars);
fun = @(x) 0;
[exitflag(id),err(id,:)] = testblock(fun,x0,LB,UB,PLB,PUB,lnZ,mubar,options,0);


% fun = @(x) sum(-abs(x)./(1:numel(x))) - sum(log(1:numel(x))) - numel(x)*log(2);

failed = 0;
fprintf('===========================================================================\n');
for i = 1:numel(txt)
    fprintf('%s:', txt{i});
    if exitflag(i) >= 0 && err(i,1) < tolerr(1) && err(i,2) < tolerr(2)
        fprintf('\tPASSED\n');
    else
        fprintf('\tFAILED\n');
        failed = failed + 1;
    end 
end
vbmc_version = vbmc('version');
fprintf('\nVBMC version: %s.\nTotal runtime: %.1f s.\tBallpark runtime on a reference computer: %.1f s.\n(Reference computer specs: %s.)\n',vbmc_version,toc(t),RunTime,specs);
fprintf('===========================================================================\n');
fprintf('\n');

if failed == 0
    display('VBMC is working correctly. See vbmc_examples.m for usage examples; check out the <a href="https://github.com/lacerbi/bads">VBMC website</a>;'); 
    display('consult the <a href="https://github.com/lacerbi/vbmc/wiki">online FAQ</a>; or type ''help vbmc'' for more information. Enjoy!');
else
    display('VBMC is not working correctly. Please check the <a href="https://github.com/lacerbi/vbmc/wiki">online FAQ</a> for more information.');
end

end

%--------------------------------------------------------------------------
function [exitflag,err] = testblock(fun,x0,LB,UB,PLB,PUB,lnZ,mubar,options,noiseflag)

if nargin < 10 || isempty(noiseflag); noiseflag = false; end

% Modify options
options.MaxFunEvals = 100;
options.Plot = false;
if noiseflag
    options.SpecifyTargetNoise = true;
end

[vp,elbo,elbo_sd,exitflag,output,optimState,stats] = ...
    vbmc(fun,x0,LB,UB,PLB,PUB,options);

vmu = vbmc_moments(vp);
err(2) = sqrt(mean((vmu - mubar).^2));    % RMSE of the posterior mean

fprintf('ELBO: %.3f +/- %.3f (lower bound on true log model evidence: %.3f), with %d function evals.\nRMSE of the posterior mean: %.3f.\n', ...
    elbo, elbo_sd, lnZ, output.funccount, err(2));
err(1) = abs(elbo - lnZ);

fprintf('\n');
end

%--------------------------------------------------------------------------
function [y,s] = noisefun(x)
    y = sum(-0.5*(x./(1:numel(x))).^2) - sum(log(1:numel(x))) - 0.5*numel(x)*log(2*pi) + randn([size(x,1),1]);
    s = 1;
end

function [y,s] = noiseuniform_fun(x)
    sigma = 1;
    n = size(x,1);
    y = sigma*randn([n,1]);
    s = sigma*ones(n,1);
end

%--------------------------------------------------------------------------

function [y,s] = cigar(x)
%CIGAR Benchmark log pdf -- cigar density.

D = size(x,2);
Mean = linspace(-0.5,0.5,D);

switch D
    case 1
        R = 10;
    case 2
        R = [0.438952107785021 -0.898510460190134;0.898510460190134 0.438952107785021];
    case 3
        R = [-0.624318398571926 -0.0583529832968072 -0.778987462379818;0.779779849779334 0.0129117551612018 -0.625920659873738;0.0465824331986329 -0.998212510399975 0.0374414342443664];
    case 4
        R = [0.530738877213611 -0.332421458771 -0.617324087669642 0.476154584925358;-0.455283846255008 -0.578972039590549 0.36136334497906 0.57177314523957;-0.340852417262338 -0.587449365484418 -0.433532840927373 -0.592260203353068;-0.628372893431681 0.457373582657411 -0.548066400653999 0.309160368031855];
    case 5
        R = [-0.435764067736038 0.484423029373161 0.0201157836447536 -0.195133090987468 0.732777208934001;-0.611399063990897 -0.741129629736756 -0.0989871013229956 0.187328571370887 0.178962612288509;-0.0340226717227732 0.234931965418636 -0.886686220684869 0.394077369749064 -0.0462601570752127;0.590564625513463 -0.400973629067102 -0.304445169104938 -0.33203681171552 0.536207298122059;0.293899205038366 0.00939792298771627 0.333012903768423 0.813194727889496 0.375967653898291];
    case 6
        R = [-0.254015072891056 -0.0684032463717124 -0.693077090686521 0.249685438636409 0.362364372413356 -0.506745230203393;-0.207777316284753 0.369766206365964 0.57903494069884 -0.0653147667578752 0.122468089523108 -0.682316367390556;-0.328071435400004 0.364091738763865 -0.166363836395589 0.380087224558118 -0.766382695507289 -0.0179075049327736;-0.867667731196277 -0.0332627076729128 0.069771482765022 -0.25333031036676 0.206274179664928 0.366678275010647;-0.0206482741639122 -0.229074515307431 -0.237811602709101 -0.777821502080957 -0.426607324185607 -0.321782626441153;-0.177201636197285 -0.820030251267824 0.308647597698151 0.346038046252171 -0.204470893859678 -0.198332931405751];
end

ell = [ones(1,D-1)/100,1];
Cov = R'*diag(ell.^2)*R;

y = mvnlogpdf(x,Mean,Cov); % + mvnlogpdf(x,priorMean,priorCov);
s = 0;

end

function y = mvnlogpdf(X, Mu, Sigma)
%MVNLOGPDF Multivariate normal log probability density function (pdf).

d = size(X,2);
X0 = bsxfun(@minus,X,Mu);
sz = size(Sigma);

% Make sure Sigma is a valid covariance matrix
[R,err] = cholcov(Sigma,0);
if err ~= 0
    error(message('stats:mvnpdf:BadMatrixSigma'));
end
% Create array of standardized data, and compute log(sqrt(det(Sigma)))
xRinv = X0 / R;
logSqrtDetSigma = sum(log(diag(R)));

% The quadratic form is the inner products of the standardized data
quadform = sum(xRinv.^2, 2);

y = -0.5*quadform - logSqrtDetSigma - d*log(2*pi)/2;

end