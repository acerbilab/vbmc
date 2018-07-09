clear;
% cd ~/Code/GP/BQR
% 
% load test_bqr_incr_samples_slice
BQR = [];BQ=[];BMC=[];MC=[];

prob_bqr_incr_samples;
max_num_samples = 200;



    samples = slicesample(prior.means, max_num_samples,...
        'pdf', p_r_fn,'width', prior.sds);
    
%     samples = (-3:0.1:3)';
%     max_num_samples = length(samples);

    q= [];
    r= [];
    
    for num_sample = 1:max_num_samples;
        % structs for q and r no longer reqd
        q = [q;q_fn(samples(num_sample,:))];
        r = [r;r_fn(samples(num_sample,:))];
    end
        
        sample_struct.samples = samples;
        sample_struct.log_r = log(r);
        sample_struct.q = q;
        
%         [r_noise_sd, r_input_scales, r_output_scale] = ...
%             hp_heuristics(samples, r, 100);

opt.optim_time = 60;

        opt.prior_mean = 'train';
        gpq = train_gp('sqdexp', 'constant', [], ...
            samples(1:num_sample,:), q, opt);
        [best_hypersample, best_hypersample_struct] = disp_hyperparams(gpq);
        
        q_gp.quad_output_scale = best_hypersample_struct.output_scale;
        q_gp.quad_input_scales = best_hypersample_struct.input_scales;
        q_gp.quad_noise_sd = best_hypersample_struct.noise_sd;
        q_gp.quad_mean = best_hypersample_struct.mean;
        
        opt.prior_mean = 0;
        gpr = train_gp('sqdexp', 'constant', [], ...
            samples(1:num_sample,:), r, opt);
        [best_hypersample, best_hypersample_struct] = disp_hyperparams(gpr);
        
        r_gp.quad_output_scale = best_hypersample_struct.output_scale;
        r_gp.quad_input_scales = best_hypersample_struct.input_scales;
        r_gp.quad_noise_sd = best_hypersample_struct.noise_sd;
        r_gp.quad_mean = best_hypersample_struct.mean;
   
        exact
        
        [BQR, ~, BQ] = ...
            predict(sample_struct, prior, r_gp, q_gp)
                
        BMC = predict_BMC(sample_struct, prior, r_gp)
        
        MC = predict_MC(sample_struct, prior)
        
       

