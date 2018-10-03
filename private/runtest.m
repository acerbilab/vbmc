function failed = runtest()
%RUNTEST Test Variational Bayesian Monte Carlo (VBMC).
%  RUNTEST executes a few runs of the VBMC inference algorithm to
%  check that it is installed correctly, and returns the number of failed
%  tests.
% 
%  See also VBMC, VBMC_EXAMPLES.

tolerr = [0.5 0.5];                     % Error tolerance

id = 1;
nvars = 6;
txt{id} = ['Test with multivariate normal distribution (D = ' num2str(nvars) ', unconstrained)'];
fprintf('%s.\n', txt{id});
x0 = -ones(1,nvars);                   % Initial point
PLB = -2*nvars; PUB = 2*nvars;
LB = -Inf; UB = Inf;
lnZ = 0; mubar = zeros(1,nvars);
fun = @(x) sum(-0.5*(x./(1:numel(x))).^2) - sum(log(1:numel(x))) - 0.5*numel(x)*log(2*pi);
[exitflag(id),err(id,:)] = testblock(fun,x0,LB,UB,PLB,PUB,lnZ,mubar);

id = 2;
nvars = 2;
txt{id} = ['Test with multivariate half-normal distribution (D = ' num2str(nvars) ', constrained)'];
fprintf('%s.\n', txt{id});
x0 = -ones(1,nvars);                   % Initial point
LB = -nvars*10; UB = 0;
PLB = -6; PUB = -0.05;
lnZ = -nvars*log(2); mubar = -2/sqrt(2*pi)*(1:nvars);
fun = @(x) sum(-0.5*(x./(1:numel(x))).^2) - sum(log(1:numel(x))) - 0.5*numel(x)*log(2*pi);
[exitflag(id),err(id,:)] = testblock(fun,x0,LB,UB,PLB,PUB,lnZ,mubar);

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
function [exitflag,err] = testblock(fun,x0,LB,UB,PLB,PUB,lnZ,mubar)

options = vbmc('defaults');             % Default options
options.MaxFunEvals = 100;
options.Plot = false;

[vp,elbo,elbo_sd,exitflag,output,optimState,stats] = ...
    vbmc(fun,x0,LB,UB,PLB,PUB,options);

vmu = vbmc_moments(vp);
err(2) = sqrt(mean((vmu - mubar).^2));    % RMSE of the posterior mean

fprintf('ELBO: %.3f +/- %.3f (lower bound on true log model evidence: %.3f), with %d function evals.\nRMSE of the posterior mean: %.3f.\n', ...
    elbo, elbo_sd, lnZ, output.funccount, err(2));
err(1) = abs(elbo - lnZ);


fprintf('\n');
end