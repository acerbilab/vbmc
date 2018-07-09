% demonstrate usage of covariance functions
%
% See also covFunctions.m.
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2014-03-21.
%                                      File automatically generated using noweb.
clear all, close all
n = 5; D = 3; x = randn(n,D); xs = randn(3,D);  % create a data set

% set up simple covariance functions
cn  = {'covNoise'}; sn = .1;  hypn = log(sn);  % one hyperparameter
cc  = {@covConst};   sf = 2;  hypc = log(sf); % function handles OK
cl  = {@covLIN};              hypl = []; % linear is parameter-free
cla = {'covLINard'}; L = rand(D,1); hypla = log(L);  % linear (ARD)
cli = {'covLINiso'}; l = rand(1);   hypli = log(l);    % linear iso
clo = {@covLINone}; ell = .9; hyplo = log(ell);  % linear with bias
cp  = {@covPoly,3}; c = 2; hypp = log([c;sf]);   % third order poly
cga = {@covSEard};   hypga = log([L;sf]);       % Gaussian with ARD
cgi = {'covSEiso'};  hypgi = log([ell;sf]);    % isotropic Gaussian
cgu = {'covSEisoU'}; hypgu = log(ell);   % isotropic Gauss no scale
cra = {'covRQard'}; al = 2; hypra = log([L;sf;al]); % ration. quad.
cri = {@covRQiso};          hypri = log([ell;sf;al]);   % isotropic
cma = {@covMaternard,5};  hypma = log([ell;sf]); % Matern class d=5
cmi = {'covMaterniso',3}; hypmi = log([ell;sf]); % Matern class d=3
cnn = {'covNNone'}; hypnn = log([L;sf]);           % neural network
cpe = {'covPeriodic'}; p = 2; hyppe = log([ell;p;sf]);   % periodic
cpn = {'covPeriodicNoDC'}; p = 2; hyppe = log([ell;p;sf]); % w/o DC
cpc = {'covCos'}; p = 2; hypcpc = log([p;sf]);         % cosine cov
cca = {'covPPard',3}; hypcc = hypm; % compact support poly degree 3
cci = {'covPPiso',2}; hypcc = hypm; % compact support poly degree 2
cgb = {@covGaboriso}; ell = 1; p = 1.2; hypgb=log([ell;p]); % Gabor
csm = {@covSM,4}; hypsm = log([w;m(:);v(:)]);    % Spectral Mixture

% set up composite i.e. meta covariance functions
csc = {'covScale',{cgu}};    hypsc = [log(3); hypgu];  % scale by 9
csu = {'covSum',{cn,cc,cl}}; hypsu = [hypn; hypc; hypl];      % sum
cpr = {@covProd,{cc,ccc}};   hyppr = [hypc; hypcc];       % product
mask = [0,1,0]; %   binary mask excluding all but the 2nd component
cma = {'covMask',{mask,cgi{:}}}; hypma = hypgi;
% isotropic periodic rational quadratic
cpi = {'covPERiso',{@covRQiso}};
% periodic Matern with ARD
cpa = {'covPERard',{@covMaternard,3}};
% additive based on SEiso using unary and pairwise interactions
cad = {'covADD',{[1,2],'covSEiso'}};

% 0) specify a covariance function
cov = cma; hyp = hypma;

% 1) query the number of parameters
feval(cov{:})

% 2) evaluate the function on x
feval(cov{:},hyp,x)

% 3) evaluate the function on x and xs to get cross-terms
kss = feval(cov{:},hyp,xs,'diag')
Ks  = feval(cov{:},hyp,x,xs)

% 4) compute the derivatives w.r.t. to hyperparameter i
i = 1; feval(cov{:},hyp,x,[],i)
