function [K,out2,out3] = wderiv_cov_fn(type,hp,flag)

T=exp(hp(3:end-1));
L=exp(hp(end));
 
if nargin<3
    flag='deriv inputs';
end

K=@(Xs1,Xs2) Kwderivrot(Xs1,Xs2,type,{T,L});

if strcmpi(flag,'deriv inputs')
    DK=@(Xs1,Xs2) DKwderivrot(Xs1,Xs2,type,{T,L});
    DDK=@(Xs1,Xs2) DDKwderivrot(Xs1,Xs2,type,{T,L});
    out2=DK;
    out3=DDK;
elseif strcmpi(flag,'deriv hyperparams')
    DK=@(Xs1,Xs2) [{zeros(size(Xs1,1),size(Xs2,1));...          % mean
                    zeros(size(Xs1,1),size(Xs2,1))};...         % logNoiseSD           
                    DTKwderivrot(Xs1,Xs2,type,{T,L});...        % logInputScales
                    {2*K(Xs1,Xs2)}];                            % logOutputScale
                      
    out2=DK;
end