function [K,out2] = MIDAS_indep_cov_fn(hp,flag)  

T=exp(hp(3));
H=exp(hp(5));

type1 = 'ratquad';

K = @(as,bs) matrify(@(al,at,bl,bt) fcov(type1,{T,H},at,bt).*(al==bl),...
    as,bs);

if nargin<2
    flag='deriv inputs';
end

if strcmpi(flag,'deriv hyperparams')               
    out2=@(Xs1,Xs2) Dhps_K(Xs1,Xs2,type1,T,H);
end

function DK = Dhps_K(Xs1,Xs2,type1,T,H)
S1 = size(Xs1,1);
S2 = size(Xs2,1);
num_hps = 5;

K_time = matrify(@(al,at,bl,bt) fcov(type1,{T,H},at,bt),Xs1,Xs2);
K_space = matrify(@(al,at,bl,bt) al==bl,Xs1,Xs2);


DK = mat2cell2d(zeros(num_hps*S1,S2),S1*ones(num_hps,1),S2);
DK(3) = cellfun(@(mat) mat.*K_space,...
    matrify(@(al,at,bl,bt) gTcov(type1,{T,H},at,bt),Xs1,Xs2),...
    'UniformOutput',false);
DK{4} = 0.*K_time; 
DK{5} = ...
    2*K_time.*K_space;
                
