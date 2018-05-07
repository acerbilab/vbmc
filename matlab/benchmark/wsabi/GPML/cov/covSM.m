function K = covSM(Q, hyp, x, z, i)

% Gaussian Spectral Mixture covariance function. The covariance function is 
% -- depending on the sign of Q -- parameterized as:
%
% SM, spectral mixture:  Q>0  => P = 1
%   k(x,z) = w'*prod(h(dxz,v,m),2), h(t,v,m) = exp(-2*pi^2*t^2*v)*cos(2*pi*t*m)
%   where dxz = ||x-z|| is the Euclidean distance between x and z,
%
% SMP, spectral mixture product: Q<0 => P = D
%   k(x,z) = prod( sum(w.*h(dxz,v,m),2) )
%   where dxz = diag(x-z) is the matrix with the difference x-z on its
%   diagonal;
%
% further, h(t,v,m) is a Gaboriso covariance function, w(DxP) are the weights
% and m(DxQ) = 1/p, v(DxQ) = (2*pi*ell)^-2 are spectral means (frequencies) and 
% variances, where p is the period and ell the length scale of the Gabor function. 
% The hyperparameters are:
%
% hyp = [ log(w(:))
%         log(m(:))
%         log(sqrt(v(:))) ]
%
% For more help on design of covariance functions, try "help covFunctions".
%
% Note that the spectral density H(s) = F[ h(t) ] of covGaboriso is given by
% H(s) = N(s|m,v)/2 + N(s|-m,v)/2 where m=1/p is the mean and v=(2*pi*ell)^-2
% is the variance of a symmetric Gaussian mixture. Hence the covGaboriso
% covariance forms a basis for the class of stationary covariances since a
% weighted sum of covGaboriso covariances corresponds to an isotropic
% location-scale mixture of a symmetric Gaussian mixture in the spectral domain.
%
% Internally, covSM constructs a weighted sum of products of 1d covGaboriso
% covariances using covMask, covProd, covScale and covSum.
%
% For more details, see 
% 1) Gaussian Process Kernels for Pattern Discovery and Extrapolation,
% ICML, 2013, by Andrew Gordon Wilson and Ryan Prescott Adams.
% 2) GPatt: Fast Multidimensional Pattern Extrapolation with Gaussian 
% Processes, arXiv 1310.5288, 2013, by Andrew Gordon Wilson, Elad Gilboa, 
% Arye Nehorai and John P. Cunningham, and
% http://mlg.eng.cam.ac.uk/andrew/pattern
%
% For Q>0, covSM corresponds to Eq. 12 in Ref (1)
% For Q<0, covSM corresponds to Eq. 14 in Ref (2) (but w here = w^2 in (14))
%
% Copyright (c) by Andrew Gordon Wilson and Hannes Nickisch, 2013-10-21.
%
% See also COVFUNCTIONS.M, COVGABORISO.M, COVGABORARD.M.

smp = Q<0; Q = abs(Q);                         % switch between covSM and covSMP
if nargin<3                                            % report no of parameters
  if smp, K = '3*D*'; else K = '(1+2*D)*'; end, K = [K,sprintf('%d',Q)]; return
end
if nargin<4, z = []; end                                   % make sure, z exists

D = size(x,2); P = smp*D+(1-smp);                               % dimensionality
lw = reshape(hyp(         1:P*Q) ,P,Q);                    % log mixture weights
lm = reshape(hyp(P*Q+    (1:D*Q)),D,Q);                     % log spectral means
ls = reshape(hyp(P*Q+D*Q+(1:D*Q)),D,Q);       % log spectral standard deviations

% In the following we construct nested cell arrays to finally obtain either
if smp % 1) the product of weighted sums of 1d covGabor covariance functions or
  fac = cell(1,D);
  for d=1:D
    add = cell(1,Q); % a) addends for weighted sum of univariate Gabor functions
    for q=1:Q, add{q} = {'covScale',{'covMask',{d,{'covGaboriso'}}}}; end
    fac{d} = {'covSum',add};                       % b) combine addends into sum
  end
  cov = {'covProd',fac};                       % c) combine factors into product
else   % 2) the weighted sum of products of 1d covGabor covariance functions.
                          % a) factors for product of univariate Gabor functions
  fac = cell(1,D); for d=1:D, fac{d} = {'covMask',{d,{'covGaboriso'}}}; end
                   % b) addends for weighted sum of multivariate Gabor functions
  add = cell(1,Q); for q=1:Q, add{q} = {'covScale',{'covProd',fac}};  end
  cov = {'covSum',add};                            % c) combine addends into sum
end
if smp   % assemble hyp; covGaboriso is parametrised using -ls-log(2*pi) and -lm
  hyp = [lw(:)'/2;          -ls(:)'-log(2*pi); -lm(:)'        ];
else
  hyp = [lw(:)'/2; reshape([-ls(:)'-log(2*pi); -lm(:)'],2*D,Q)];
end

if nargin<5                                       % evaluation of the covariance
  K = feval(cov{:},hyp(:),x,z);
else
  % We compute the indices j in the new hyperparameter vector hyp. The
  % correction constants c are needed because some hyperparameters in hyp are
  % powers of hyperparameters accepted by covSM.
  if i<=P*Q                                               % derivatives w.r.t. w
    c =  0.5;
    if smp, j = 1+3*(i-1); else j = 1+(i-1)*(2*D+1); end
  elseif i<=(P+D)*Q                                       % derivatives w.r.t. m
    c = -1.0; j = i-P*Q;
    if smp, j = 3+3*(j-1); else j = 2*j+floor((j-1)/D)+1; end
  elseif i<=(P+2*D)*Q                                     % derivatives w.r.t. v
    c = -1.0; j = i-(P+D)*Q;
    if smp, j = 2+3*(j-1); else j = 2*j+floor((j-1)/D); end
  else
    error('Unknown hyperparameter')
  end
  K = c*feval(cov{:},hyp(:),x,z,j);
end