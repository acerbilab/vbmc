function [history,post,algoptions] = infalgo_laplace(algo,algoset,probstruct)

algoptions.MaxFunEvals = probstruct.MaxFunEvals;

% Options from current problem
switch algoset
    case {0,'debug'}; algoset = 'debug'; algoptions.Debug = 1; algoptions.Plot = 'scatter';
    case {1,'base'}; algoset = 'base';           % Use defaults        
    otherwise
        error(['Unknown algorithm setting ''' algoset ''' for algorithm ''' algo '''.']);
end

PLB = probstruct.PLB;
PUB = probstruct.PUB;
LB = probstruct.LB;
UB = probstruct.UB;
x0 = probstruct.InitPoint;
D = size(x0,2);

% Add log prior to function evaluation 
probstruct.AddLogPrior = true;

% Compute Laplace approximation (assume mode is already given)
%--------------------------------------------------------------------------
algo_timer = tic;

% SaveTicks = probstruct.SaveTicks(probstruct.SaveTicks <= algoptions.MaxFunEvals);
fun = @(x_) infbench_func(x_,probstruct);
x0 = probstruct.Post.Mode;
y0 = fun(x0);

% Hessian matrix
A = -hessian(fun,x0);
lnZ = y0 + 0.5*D*log(2*pi) - 0.5*log(det(A));
Sigma = inv(A);

TotalTime = toc(algo_timer);
%--------------------------------------------------------------------------

post = [];

history = infbench_func(); % Retrieve history
% history.scratch.output = output;
history.TotalTime = TotalTime;
history.Output.X = [];
history.Output.y = [];

post.lnZ = lnZ;
post.lnZ_var = 0;
post.Mean = x0;
post.Cov = Sigma;
post.Mode = x0;

[kl1,kl2] = mvnkl(post.Mean,post.Cov,probstruct.Post.Mean,probstruct.Post.Cov);
post.gsKL = 0.5*(kl1 + kl2);

end