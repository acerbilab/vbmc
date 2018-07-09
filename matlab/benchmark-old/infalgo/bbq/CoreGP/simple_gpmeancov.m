function [f,C] = simple_gpmeancov(XStar,XData,YData,fnName,hps)
% for a gp with one-dimensional output Y - for now I'm just going to be using
% it for modelling a standard deviation, so should be OK.
% Actually I will also assume a one dimensional X (e.g. time) for now as
% well)
% hps = {inputscale, outputscale, mean, noiseSD}

if nargin<5
    hps={1,1,0,0};
    if nargin<4
        fnName = 'sqdexp';
    end
end

if length(hps)<3
    Mu=0;
else
    Mu = hps{3};
end

if length(hps)<4
    noiseSD = 0;
else
    noiseSD = hps{4};
end

Kfn = @(varargin) fcov(fnName,hps([1 2]),varargin{:});
vecK = @(Xs1,Xs2) fcov(fnName,hps([1 2]),Xs1,Xs2,'vector style');
%DKfn = @(varargin) gcov(fnName,hps,varargin{:});

K = @(Xs1,Xs2) matrify(Kfn,Xs1,Xs2);
%DK = @(Xs1,Xs2) matrify(DKfn,Xs1,Xs2);

lowr.UT = true;
lowr.TRANSA = true;
uppr.UT = true;

NData = size(XData, 1);

cholK = chol(K(XData, XData) + noiseSD^2 * eye(NData));

datahalf = linsolve(cholK, YData - Mu, lowr);
datatwothirds = linsolve(cholK, datahalf, uppr);
      
f = Mu + K(XStar,XData)*datatwothirds;     % Compute the objective function value at XStar
if nargout>1
KStarData=K(XStar,XData);
%         Kterm=linsolve(cholK,linsolve(cholK,KStarData',lowr),uppr);
%         C = K(XStar,XStar)-KStarData*Kterm;
        
        F = linsolve(cholK,KStarData',lowr);
        C = vecK(XStar,XStar)-sum(F.^2)'; 
end