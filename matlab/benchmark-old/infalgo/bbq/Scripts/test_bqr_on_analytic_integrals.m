num_dims = 10;
num_trials = 10;
num_samples = 10;

exact = nan(num_dims, num_trials);
BQR = nan(num_dims, num_trials);
BQ = nan(num_dims, num_trials);
BMC = nan(num_dims, num_trials);
MC = nan(num_dims, num_trials);

for dims = 1:num_dims
    fprintf('dim = %g\n', dims);
    for trial = 1:num_trials;
        fprintf('.');
        
        prior = struct();
        q = struct();
        r = struct();
        opt = struct();
        r_gp = struct();
        sample_struct = struct();
    
        prior.means = zeros(dims,1);
        prior.sds = ones(dims,1);

        log_q_mean = randn(dims,1);
        log_r_mean = randn(dims,1);

        q.mean = exp(log_q_mean);
        r.mean = exp(log_r_mean);

        log_q_R = randn(dims,dims);
        log_r_R = randn(dims,dims);
        q_R = exp(log_q_R);
        r_R = exp(log_r_R);

        q.cov = q_R' * q_R;
        r.cov = r_R' * r_R;

        %exact(dims, trial) = predict_exact(q, r, prior);
        
        L = diag(prior.sds);
        R = chol(r.cov + L);
        two_thirds = solve_chol(R, L)';
        samples = ...
            mvnrnd((prior.means + two_thirds * (r.mean - prior.means))', ...
                    L - two_thirds * L, num_samples);
            
        % structs for q and r no longer reqd
        q = mvnpdf(samples, q.mean', q.cov);
        r = mvnpdf(samples, r.mean', r.cov);
        
        sample_struct.samples = samples;
        sample_struct.log_r = log(r);
        sample_struct.q = q;
        
        
        [r_noise_sd, r_input_scales, r_output_scale] = ...
            hp_heuristics(samples, r, 100);
        
        r_gp.quad_output_scale = r_output_scale;
        r_gp.quad_input_scales = 10*r_input_scales;
        r_gp.quad_noise_sd = r_noise_sd;
   
        BQR(dims, trial) = predict(sample_struct, prior, r_gp);
        
        opt.no_adjustment = true;
        BQ(dims, trial) = predict(sample_struct, prior, r_gp, opt);
        
        BMC(dims, trial) = predict_BMC(sample_struct, prior, r_gp);
        
        MC(dims, trial) = predict_MC(sample_struct, prior, r_gp);
    end
    fprintf('\n');
end

save test_bqr_on_analytic_integrals