%cd ~/Code/gp-code-osborne/

max_num_samples = 200;
max_trials = 100;



% matlabpool close force
% matlabpool open

prior = struct();
q = struct();
l = struct();
opt = struct();
l_gp = struct();
sample_struct = struct();


        opt.print = false;
        opt.optim_time = 10;
        opt.num_hypersamples = 25;
        opt.noiseless = true;

prior.means = 0;
prior.sds = 1;
prior.mean = 0;
prior.covariance = 1;

qe(1).mean = 0.2;
qe(1).cov = 0.1;
qe(1).weight = -0.4;

qe(2).mean = 0;
qe(2).cov = 50;
qe(2).weight = 10;

qe(3).mean = -0.2;
qe(3).cov = 0.05;
qe(3).weight = 0.2;

qe(4).mean = 1.2;
qe(4).cov = 0.05;
qe(4).weight = 0.3;

le(1).mean = -1;
le(1).cov = 0.25;
le(1).weight = 0.4;

le(2).mean = 0.5;
le(2).cov = 25;
le(2).weight = 1;

le(3).mean = 2;
le(3).cov = 0.5;
le(3).weight = 0.1;

exact = predict_exact(qe, le, prior);


p_fn = @(x) normpdf(x, prior.means, prior.sds);
l_fn = @(x) sum([le(:).weight].*normpdf(x, [le(:).mean], sqrt([le(:).cov])));
q_fn = @(x) sum([qe(:).weight].*normpdf(x, [qe(:).mean], sqrt([qe(:).cov])));

log_l_fn = @(x) log(sum([le(:).weight].*normpdf(x, [le(:).mean], sqrt([le(:).cov]))));


d_p_fn = @(x) normpdf(x, prior.means, prior.sds) ...
            * (prior.means - x)/prior.sds;
d_l_fn = @(x) sum([le(:).weight].*(...
                normpdf(x, [le(:).mean], sqrt([le(:).cov])) ...
                .* ([le(:).mean] - x)./sqrt([le(:).cov])));
d_q_fn = @(x) sum([qe(:).weight].*(...
                normpdf(x, [qe(:).mean], sqrt([qe(:).cov])) ...
                .* ([qe(:).mean] - x)./sqrt([qe(:).cov])));
            
            clf;ezplot(q_fn,[-3,3]);hold on;ezplot(l_fn,[-3,3]);axis tight;
            
            
p_l_fn = @(x) p_fn(x) * l_fn(x);


