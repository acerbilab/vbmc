%cd ~/Code/gp-code-osborne/

max_num_samples = 200;
max_trials = 100;



% matlabpool close force
% matlabpool open

prior = struct();
q = struct();
r = struct();
opt = struct();
r_gp = struct();
sample_struct = struct();


        opt.print = false;
        opt.optim_time = 10;
        opt.num_hypersamples = 25;
        opt.noiseless = true;

prior.means = 0;
prior.sds = 1;

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

re(1).mean = -1;
re(1).cov = 0.25;
re(1).weight = 0.4;

re(2).mean = 0.5;
re(2).cov = 25;
re(2).weight = 1;

re(3).mean = 2;
re(3).cov = 0.5;
re(3).weight = 0.1;

exact = predict_exact(qe, re, prior);


p_fn = @(x) normpdf(x, prior.means, prior.sds);
r_fn = @(x) sum([re(:).weight].*normpdf(x, [re(:).mean], sqrt([re(:).cov])));
q_fn = @(x) sum([qe(:).weight].*normpdf(x, [qe(:).mean], sqrt([qe(:).cov])));

d_p_fn = @(x) normpdf(x, prior.means, prior.sds) ...
            * (prior.means - x)/prior.sds;
d_r_fn = @(x) sum([re(:).weight].*(...
                normpdf(x, [re(:).mean], sqrt([re(:).cov])) ...
                .* ([re(:).mean] - x)./sqrt([re(:).cov])));
d_q_fn = @(x) sum([qe(:).weight].*(...
                normpdf(x, [qe(:).mean], sqrt([qe(:).cov])) ...
                .* ([qe(:).mean] - x)./sqrt([qe(:).cov])));
            
            clf;ezplot(q_fn,[-3,3]);hold on;ezplot(r_fn,[-3,3]);axis tight;
            
            
p_r_fn = @(x) p_fn(x) * r_fn(x);


