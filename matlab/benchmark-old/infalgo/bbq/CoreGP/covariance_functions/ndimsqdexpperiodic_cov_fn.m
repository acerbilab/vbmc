function [K,DK,DDK] = ndimsqdexpperiodic_cov_fn(hp,n)
	
T=exp(hp(1:n));
L=exp(hp(n+1));

K=@(Xs1,Xs2) matrify(@(varargin)(fcov('sqdexp',{T,L},varargin{:}) + ...
  fcov({'sqdexp', 'periodic'},{T,L},varargin{:})),Xs1,Xs2);
DK=@(Xs1,Xs2) matrify(@(varargin)(cellfun(@plus, gcov('sqdexp',{T,L},varargin{:}), ...
  gcov({'sqdexp', 'periodic'},{T,L},varargin{:}),'UniformOutput',false)),Xs1,Xs2);
DDK=@(Xs1,Xs2) matrify(@(varargin)(cellfun(@plus, Hcov('sqdexp',{T,L},varargin{:}), ...
  Hcov({'sqdexp', 'periodic'},{T,L},varargin{:}),'UniformOutput',false)),Xs1,Xs2);
