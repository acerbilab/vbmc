N = 10;

addpath(genpath('~/Code/convex_programming/'))

% create a random 1000-dim covariance matrix with maximum eigenvalue 1
R = rand(N);
Sigma = R' * R;
max_eig = eigs(Sigma, 1);
Sigma = Sigma ./ max_eig;

mu = zeros(N, 1);
l = zeros(N, 1);
u = inf(N, 1);

% in particular, if opt.data is supplied, the locations for the Gaussian
% convolution observations are supplied.
% opt.data(i).m represents the mean of a Gaussian, 
% opt.data(i).m V the diagonal of its diagonal covariance.

% no data test
% =================================================
opt.data = [];

[ m_Z, sd_Z, data ] = mvncdf_bq( l, u, mu, Sigma, opt )

% two datum test
% =================================================
for i = 1:1
    opt.data(i).m = 0.5*rand(N, 1);
    opt.data(i).V = 0.05*rand(N, 1);
end

[ m_Z, sd_Z, data ] = mvncdf_bq( l, u, mu, Sigma, opt )