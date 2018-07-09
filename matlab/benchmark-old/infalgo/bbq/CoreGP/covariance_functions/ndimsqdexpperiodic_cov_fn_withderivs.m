function [K,out2,out3] = ndimsqdexpperiodic_cov_fn_withderivs(hp,n,flag)

T=exp(hp(1:n));
L=exp(hp(n+1));

type='sqdexp';

if nargin<3
    flag='deriv inputs';
end

K=@(Xs1,Xs2) Kwderivrot(Xs1,Xs2,type,{T,L})+Kwderivrot(Xs1,Xs2,{type,'periodic'},{T,L});

if strcmpi(flag,'deriv inputs')
    DK=@(Xs1,Xs2) cellfun(@plus,...
            DKwderivrot(Xs1,Xs2,type,{T,L}),...
            DKwderivrot(Xs1,Xs2,{type,'periodic'},{T,L}),...
            'UniformOutput',false);
    DDK=@(Xs1,Xs2) cellfun(@plus,...
            DDKwderivrot(Xs1,Xs2,type,{T,L}),...
            DDKwderivrot(Xs1,Xs2,{type,'periodic'},{T,L}),...
            'UniformOutput',false);
    out2=DK;
    out3=DDK;
elseif strcmpi(flag,'deriv hyperparams')
    DK=@(Xs1,Xs2) [cellfun(@plus,...
                        DTKwderivrot(Xs1,Xs2,type,{T,L}),...
                        DTKwderivrot(Xs1,Xs2,{type,'periodic'},{T,L}),...
                        'UniformOutput',false);                 % logInputScales
                    {2*K(Xs1,Xs2)};                             % logOutputScale
                    {zeros(size(Xs1,1),size(Xs2,1));...          % mean
                    zeros(size(Xs1,1),size(Xs2,1))}];             % logNoiseSD    
                           
    out2=DK;
end
