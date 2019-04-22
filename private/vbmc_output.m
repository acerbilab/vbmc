function output = vbmc_output(elbo,elbo_sd,optimState,msg,stats,idx_best)
%VBMC_OUTPUT Create OUTPUT struct for VBMC.

output.function = func2str(optimState.fun);
if all(isinf(optimState.LB)) && all(isinf(optimState.UB))
    output.problemtype = 'unconstrained';
else
    output.problemtype = 'boundconstraints';
end
output.iterations = optimState.iter;
output.funccount = optimState.funccount;
output.bestiter = idx_best;
output.trainsetsize = stats.Neff(idx_best);
output.components = stats.vpK(idx_best);
output.rindex = stats.rindex(idx_best);
if stats.stable(idx_best)
    output.convergencestatus = 'probable';
else
    output.convergencestatus = 'no';
end
output.overhead = NaN;
output.rngstate = rng;
output.algorithm = 'Variational Bayesian Monte Carlo';
output.message = msg;

output.elbo = elbo;
output.elbo_sd = elbo_sd;

end