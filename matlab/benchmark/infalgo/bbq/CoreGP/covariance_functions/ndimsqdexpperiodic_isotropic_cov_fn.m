function [K,DK,DDK] = ndimsqdexpperiodic_isotropic_cov_fn(hp,n)

T=exp(hp(1)) * ones(1,n);
L=exp(hp(2));

T2=exp(hp(3)) * ones(1,n);
L2=exp(hp(4));

K=@(Xs1,Xs2) matrify(@(varargin)(fcov('sqdexp',{T,L},varargin{:}) + ...
  fcov({'sqdexp', 'periodic'},{T2,L2},varargin{:})),Xs1,Xs2);
DK=@(Xs1,Xs2) matrify(@(varargin)(cellfun(@plus, gcov('sqdexp',{T,L},varargin{:}), ...
  gcov({'sqdexp', 'periodic'},{T2,L2},varargin{:}),'UniformOutput',false)),Xs1,Xs2);
DDK=@(Xs1,Xs2) matrify(@(varargin)(cellfun(@plus, Hcov('sqdexp',{T,L},varargin{:}), ...
  Hcov({'sqdexp', 'periodic'},{T2,L2},varargin{:}),'UniformOutput',false)),Xs1,Xs2);
