clear;

D = 1;
num_convs = 100;

for i = 1:num_convs
    opt.data(i).m = rand(D,1);
    opt.data(i).V = rand(D,1);
end

mu = rand(D,1);
Sigma = rand(D);
Sigma = Sigma'*Sigma;
l = rand(D,1);
u = rand(D,1)+l;


real_ans = mvncdf(l, u, mu, Sigma)
[ m_Z, sd_Z ] = mvncdf_bq( l, u, mu, Sigma, opt )

