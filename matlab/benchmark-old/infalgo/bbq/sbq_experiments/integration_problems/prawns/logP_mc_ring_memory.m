function [logP, samples] = logP_mc_ring_memory(theta, direction, N, modelidx, type)

if nargin < 5
    type = 1;
    
    if nargin <4
        modelidx = 1;
    end
end

%downsample inputs, coreelation length is ~10 frames
for i = 1:numel(theta)
    theta = theta(:, 1:2:end);
    direction = direction(:, 1:2:end);
end

logP = zeros(N, 1, 'double');
samples = zeros(N, 6, 'double');

priormin = [0, 1, -2, -2, 0, -7.5];
priormax = [pi, 5, 2, 2, 1, -7.49];
priorrange = priormax-priormin;

switch modelidx
    case 0
        log_l_pdf = @(x) logP_ring_null(theta, direction, x(1), x(2), x(3:4), x(5), x(6));
    case 1
        log_l_pdf = @(x) logP_ring_mf(theta, direction, x(1), x(2), x(3:4), x(5), x(6));
    case 2
        log_l_pdf = @(x) logP_ring_topo(theta, direction, x(1), x(2), x(3:4), x(5), x(6));
    case 3
        log_l_pdf = @(x) logP_ring_R(theta, direction, x(1), x(2), x(3:4), x(5), x(6));
    case 4
        log_l_pdf = @(x) logP_ring_R_2ways(theta, direction, x(1), x(2), x(3:4), x(5), x(6));
    case 5
        log_l_pdf = @(x) logP_ring_R_ahead(theta, direction, x(1), x(2), x(3:4), x(5), x(6));
    case 6
        log_l_pdf = @(x) logP_ring_R_ahead2ways(theta, direction, x(1), x(2), x(3:4), x(5), x(6));       
    case 7
        log_l_pdf = @(x) logP_ring_memory(theta, direction, x(1), x(2), x(3:4), x(5), x(6));
    case 8
        log_l_pdf = @(x) logP_ring_memory_2ways(theta, direction, x(1), x(2), x(3:4), x(5), x(6));     
    case 9
        log_l_pdf = @(x) logP_ring_memory_ahead(theta, direction, x(1), x(2), x(3:4), x(5), x(6));
    case 10
        log_l_pdf = @(x) logP_ring_memory_ahead2ways(theta, direction, x(1), x(2), x(3:4), x(5), x(6));   
end

log_prior_pdf = @(x) sum(log(unifpdf(x, priormin, priormax)));
prior_rnd = @(x) unifrnd(priormin, priormax);

log_pdf = @(x) log_l_pdf(x) + log_prior_pdf(x);

if type == 1
    for i = 1:N
        samples(i, :) = prior_rnd(1);
        
    end
    parfor s = 1:N
        
        logP(s) = log_l_pdf(samples(s, :));   
        
    end
elseif type ==2
    propM = diag(priorrange/20).^2;
    prop_rnd = @(x) mvnrnd(x, propM);
    start = prior_rnd(1);
    samples = mhsample(start, N, 'logpdf', log_pdf, 'proprnd', prop_rnd, 'symmetric', 1, 'burnin', 0);
elseif type ==3
    start = prior_rnd(1);
    samples = slicesample(start, N, 'logpdf', log_pdf);
end




