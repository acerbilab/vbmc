function [K,DK,DDK] = ndimsqdexp_isotropic_cov_fn(hp,n,flag)
 	
T=exp(hp(1)) * ones(1,n);
L=exp(hp(2));
	
K=@(Xs1,Xs2) matrify(@(varargin) fcov('sqdexp',{T,L},varargin{:}),Xs1,Xs2);
DK=@(Xs1,Xs2) matrify(@(varargin) gcov('sqdexp',{T,L},varargin{:}),Xs1,Xs2);
DDK=@(Xs1,Xs2) matrify(@(varargin) Hcov('sqdexp',{T,L},varargin{:}),Xs1,Xs2);


