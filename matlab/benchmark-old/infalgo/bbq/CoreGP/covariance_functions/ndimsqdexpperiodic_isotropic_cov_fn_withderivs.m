function [K,out2,out3] = ndimsqdexpperiodic_isotropic_cov_fn_withderivs(hp,n,flag)

T=exp(hp(1)) * ones(1,n);
L=exp(hp(2));

T2=exp(hp(3)) * ones(1,n);
L2=exp(hp(4));

if nargin<3
    flag='deriv inputs';
end

type='sqdexp';

K=@(Xs1,Xs2) Kwderivrot(Xs1,Xs2,type,{T,L})+Kwderivrot(Xs1,Xs2,{type,'periodic'},{T2,L2});

if strcmpi(flag,'deriv inputs')
    DK=@(Xs1,Xs2) cellfun(@plus,...
            DKwderivrot(Xs1,Xs2,type,{T,L}),...
            DKwderivrot(Xs1,Xs2,{type,'periodic'},{T2,L2}),...
            'UniformOutput',false);
    DDK=@(Xs1,Xs2) cellfun(@plus,...
            DDKwderivrot(Xs1,Xs2,type,{T,L}),...
            DDKwderivrot(Xs1,Xs2,{type,'periodic'},{T2,L2}),...
            'UniformOutput',false);
    out2=DK;
    out3=DDK;
elseif strcmpi(flag,'deriv hyperparams')
    DK=@(Xs1,Xs2) {DInputHp(Xs1,Xs2,type,{T,L});...                         % logInputScale (for non-periodic term)
                    2*Kwderivrot(Xs1,Xs2,type,{T,L});...                  % logOutputScale (for non-periodic term)
                    DInputHp(Xs1,Xs2,{type,'periodic'},{T2,L2});...         % logInputScale (for periodic term)
                    2*Kwderivrot(Xs1,Xs2,{type,'periodic'},{T2,L2});       % logOutputScale (for periodic term)
                    zeros(size(Xs1,1),size(Xs2,1));...                      % mean
                    zeros(size(Xs1,1),size(Xs2,1))};                       % logNoiseSD                                       
    out2=DK;
end

function out=DInputHp(varargin)
out=DTKwderivrot(varargin{:});
out=sum(cat(3,out{:}),3);