function [f,g] = simple_gpmean(XStar,XData,YData,fnName,hps)
% for a gp with one-dimensional output Y - for now I'm just going to be using
% it for modelling a standard deviation, so should be OK.
% Actually I will also assume a one dimensional X (e.g. time) for now as
% well)
Mu=0;

Kfn = @(varargin) fcov(fnName,hps,varargin{:});
DKfn = @(varargin) gcov(fnName,hps,varargin{:});

K = @(Xs1,Xs2) matrify(Kfn,Xs1,Xs2);
DK = @(Xs1,Xs2) matrify(DKfn,Xs1,Xs2);

datatwothirds=K(XData,XData)\(YData-Mu);

f = Mu + K(XStar,XData)*datatwothirds;     % Compute the objective function value at XStar
if nargout>1
g = cellfun(@(DKmat) DKmat*datatwothirds,DK(XStar,XData),'UniformOutput',false);  % Gradient of the function evaluated at XStar
end