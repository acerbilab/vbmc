function [K,out2,out3] = ndimsqdexp_isotropic_cov_fn_withderivs(hp,n,flag)

T=exp(hp(1)) * ones(1,n);
L=exp(hp(2));

type='sqdexp';

if nargin<3
    flag='deriv inputs';
end



%K=@(Xs1,Xs2) Kwderivrot(Xs1,Xs2,type,{T,L});
K=@(Xs1,Xs2) matrify(@(varargin) fcov('sqdexp',{T,L},varargin{:}),Xs1,Xs2);

if strcmpi(flag,'deriv inputs')
  DK=@(Xs1,Xs2) matrify(@(varargin) gcov('sqdexp',{T,L},varargin{:}),Xs1,Xs2);
  DDK=@(Xs1,Xs2) matrify(@(varargin) Hcov('sqdexp',{T,L},varargin{:}),Xs1,Xs2);
%    DK=@(Xs1,Xs2) DKwderivrot(Xs1,Xs2,type,{T,L});
%    DDK=@(Xs1,Xs2) DDKwderivrot(Xs1,Xs2,type,{T,L});
    out2=DK;
    out3=DDK;
elseif strcmpi(flag,'deriv hyperparams')
    DK=@(Xs1,Xs2) {DInputHp(Xs1,Xs2,type,{T,L});...        % logInputScales
                    2*K(Xs1,Xs2);                              % logOutputScale
                    zeros(size(Xs1,1),size(Xs2,1));...          % mean
                    zeros(size(Xs1,1),size(Xs2,1))};         % logNoiseSD                                       
    out2=DK;
end

function out=DInputHp(Xs1,Xs2,type,hps)
out = matrify(@(varargin) gTcov(type,hps,varargin{:}),Xs1,Xs2);
%out=DTKwderivrot(varargin{:});
out=sum(cat(3,out{:}),3);
