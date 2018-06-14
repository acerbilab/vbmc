cd ~/Code/gp-code-osborne/
% % addpath(genpath('~/Code/gp-code-osborne/'))
% % addpath ~/Code/lightspeed
% % addpath(genpath('~/Code/Utils/'))
% % rmpath ~/Code/CoreGP
% % rmpath ~/Code/BQR
% 
% 
% clear
% series=load('~/Code/gp-code-osborne/ChrisCandidates/data337.txt');
% 
% series(1,:) = series(1,:) - min(series(1,:));
% series = series';
% num_series = length(series);
% 
% X = series(:,1);
% y = series(:,2);
% 
% NN = 5;
% %try to get NN points per day
% step = ceil(num_series/range(X)/NN);
% 
% f_data = 1:step:num_series;
% f_star = ceil(step/2):step:num_series;
% num_data = length(f_data);
% num_star = length(f_star);
% 
% y = fastsmooth(y,step);
% 
% X_data = X(f_data);
% y_data = y(f_data);
% 
% X_star = X(f_star);
% y_star = y(f_star);
% 
% 
% opt.print = false;
% opt.optim_time = 30;
% opt.num_hypersamples = 25;
% opt.noiseless = true;
% 
% max_num_samples = 500;
% 
% %load data
% 
% %k(t,t') = A * exp[ - sin^2 (pi (t-t') / P) / (2 L1^2) ...
% % - (t-t')^2 / (2 L2^2) ] + sigma^2 * delta (t-t') 
% 
% [est_noise_sd,est_input_scales,est_output_scale] = ...
%       hp_heuristics(X_data,y_data-mean(y_data),100);
% mean_y = mean(y_data);
% 
% gp.hyperparams(1) = ...
%                 struct('name','logNoiseSD',...
%                 'priorMean',log(est_noise_sd),...
%                 'priorSD',0.5);
% gp.hyperparams(2) = ...
%                 struct('name','logInputScale',...
%                 'priorMean',log(10*est_input_scales),...
%                 'priorSD',0.5);
% gp.hyperparams(3) = ...
%                 struct('name','logOutputScale',...
%                 'priorMean',log(est_output_scale),...
%                 'priorSD',0.5);
% gp.hyperparams(4) = ...
%                 struct('name','logPeriod',...
%                 'priorMean',log(7),...
%                 'priorSD',0.5);
% gp.hyperparams(5) = ...
%                 struct('name','logRoughness',...
%                 'priorMean',log(10*est_input_scales),...
%                 'priorSD',0.5);
% 
% % gp.hyperparams(1) = ...
% %                 struct('name','logNoiseSD',...
% %                 'priorMean',0,...
% %                 'priorSD',2);
% % gp.hyperparams(2) = ...
% %                 struct('name','logInputScale',...
% %                 'priorMean',0,...
% %                 'priorSD',2);
% % gp.hyperparams(3) = ...
% %                 struct('name','logOutputScale',...
% %                 'priorMean',0,...
% %                 'priorSD',2);
% % gp.hyperparams(4) = ...
% %                 struct('name','logPeriod',...
% %                 'priorMean',0,...
% %                 'priorSD',2);
% % gp.hyperparams(5) = ...
% %                 struct('name','logRoughness',...
% %                 'priorMean',0,...
% %                 'priorSD',2);
% gp.hyperparams(6) = ...
%                 struct('name','MeanConst',...
%                 'priorMean',mean_y,...
%                 'priorSD',0.5);
%             
% hps_struct = set_hps_struct(gp);
%             
% gp.covfn = @(flag) flux_cov_fn(hps_struct,flag);
% 
% % we do marginalise over priorMean
% active_hp_inds = 1:numel(gp.hyperparams);
% gp.active_hp_inds = active_hp_inds;
% 
% prior_means = vertcat(gp.hyperparams(active_hp_inds).priorMean);
% prior_sds = vertcat(gp.hyperparams(active_hp_inds).priorSD);
% prior.means = prior_means;
% prior.sds = prior_sds;
% 
% log_p_fn = @(x) logmvnpdf(x, prior_means', diag(prior_sds.^2));
% log_r_fn = @(x) log_gp_lik(x, X_data, y_data, gp);
% 
% 
% 
% log_p_r_fn = @(x) log_p_fn(x) + log_r_fn(x);
%         
% %for trial = 1:num_trials
% 
% % Problem.f = @(x) -log_p_r_fn(x');
% % opts.maxevals = 100;
% % [a,b] = Direct(Problem, [prior_means - prior_sds,prior_means + prior_sds],opts)
% 
%     samples = slicesample(prior_means', max_num_samples,...
%         'logpdf', log_p_r_fn,'width', prior_sds');
% 
% 
% 
% 
% 
% % neg_log_rp = @(x) - log_r_fn(x) - log_p_fn(x);
% % d_neg_log_rp = @(x) - d_r_fn(x)/r_fn(x) - d_p_fn(x)/p_fn(x);
% % 
% % hmc2('state', 0);
% % hmc_options = struct('nsamples',max_num_samples*3,...
% %         'nomit',0,'display',0,'stepadj',prior.sds);
% % hmc_options = hmc2_opt(hmc_options);
% % 
% % samples = ...
% %     hmc2(neg_log_rp, prior.means, hmc_options, d_neg_log_rp);
% % [unique_samples, inds] = unique(samples);
% % samples = samples(sort(inds));
% % samples = samples(1:max_num_samples);
% 
% 
% for i = 1:max_num_samples
%     gp.hypersamples(i).hyperparameters(active_hp_inds) = samples(i,:);
%     gp.hypersamples(i).hyperparameters(hps_struct.MeanConst) = mean_y;
% end
% gp.grad_hyperparams = false;
% gp = revise_gp(X_data, y_data, gp, 'overwrite');
% 
% log_r = vertcat(gp.hypersamples.logL);
% 
% % hold on;
% % plot(samples(4,:), exp(log_r-max(log_r)),'.')
% 
% mean_y = nan(num_star, max_num_samples);
% var_y = nan(num_star, max_num_samples);
% for hs = 1:max_num_samples
%     [mean_y(:, hs), var_y(:, hs)] = ...
%         posterior_gp(X_star,gp,hs,'var_not_cov');
% end
% 
% mean_y = mean_y';
% var_y = var_y';
% 
% qd = mean_y;
% qdd = var_y + mean_y.^2;

load prob_bqr_on_flux_prediction

qd_gp_mean = qd(1,:);
qdd_gp_mean = qdd(1,:);

gpr = [];
gpqd = [];
gpqdd = [];

ML_mean = nan(num_star,max_num_samples);
MC_mean = nan(num_star,max_num_samples);
BMC_mean = nan(num_star,max_num_samples);
BQ_mean = nan(num_star,max_num_samples);
BQR_mean = nan(num_star,max_num_samples);

ML_sd = nan(num_star,max_num_samples);
MC_sd = nan(num_star,max_num_samples);
BMC_sd = nan(num_star,max_num_samples);
BQ_sd = nan(num_star,max_num_samples);
BQR_sd = nan(num_star,max_num_samples);

perf_ML = nan(1,max_num_samples);
perf_MC = nan(1,max_num_samples);
perf_BMC = nan(1,max_num_samples);
perf_BQ = nan(1,max_num_samples);
perf_BQR = nan(1,max_num_samples);


warning('off','revise_gp:small_num_data');
 warning('off','train_gp:insuff_time');

for i = 1:max_num_samples

      samples_i = samples(1:i, :);
        log_r_i = log_r(1:i, :);
        log_r_i = log_r_i - max(log_r_i);
        r_i = exp(log_r_i);
        qd_i = qd(1:i, :);
        qdd_i = qdd(1:i, :);




        sample_struct = struct();
        sample_struct.samples = samples_i;
        sample_struct.log_r = log_r_i;
        sample_struct.qd = qd_i;
        sample_struct.qdd = qdd_i;     

        opt.optim_time = 30;
        opt.active_hp_inds = 2:8;
        opt.prior_mean = 0;
        opt.num_hypersamples = 10;
        
        gpr = train_gp('sqdexp', 'constant', gpr, ...
            samples_i, r_i, opt);
        [best_hypersample, best_hypersample_struct] = disp_hyperparams(gpr);

        r_gp.quad_output_scale = best_hypersample_struct.output_scale;
        r_gp.quad_input_scales = best_hypersample_struct.input_scales;
        r_gp.quad_noise_sd = best_hypersample_struct.noise_sd;
        r_gp.quad_mean = 0;
        
        [max_log_r, max_ind] = max(log_r_i);
        qd_gp_mean = qd_i(max_ind,:);%sum(bsxfun(@times,qd_i, r_i/sum(r_i)),1);
        qdd_gp_mean = qdd_i(max_ind,:);%sum(bsxfun(@times,qdd_i, r_i/sum(r_i)),1);

        
        % rotate through the columns of qd_i
        column = mod(i-1,num_star)+1;
        
        opt.prior_mean = qd_gp_mean(column);
        gpqd = train_gp('sqdexp', 'constant', gpqd, ...
            samples_i, qd_i(:,column), opt);
        [best_hypersample, best_hypersample_struct] = disp_hyperparams(gpqd);

        qd_gp.quad_output_scale = best_hypersample_struct.output_scale;
        qd_gp.quad_input_scales = best_hypersample_struct.input_scales;
        qd_gp.quad_noise_sd = best_hypersample_struct.noise_sd;
        qd_gp.quad_mean = qd_gp_mean;

        
        opt.prior_mean = qdd_gp_mean(column);
        gpqdd = train_gp('sqdexp', 'constant', gpqdd, ...
            samples_i, qdd_i(:,column), opt);
        [best_hypersample, best_hypersample_struct] = disp_hyperparams(gpqdd);

        qdd_gp.quad_output_scale = best_hypersample_struct.output_scale;
        qdd_gp.quad_input_scales = best_hypersample_struct.input_scales;
        qdd_gp.quad_noise_sd = best_hypersample_struct.noise_sd;
        qdd_gp.quad_mean = qdd_gp_mean;
        

             
%             opt.optim_time = 2;
%             opt.active_hp_inds = [];
%             opt.prior_mean = 'train';
%             opt.num_hypersamples = 1;
% 
%             parfor star_ind = 1:num_star
% 
%                 gpqd_star = gpqd;
%                 for sample_ind = 1:numel(gpqd_star.hypersamples)
%                     gpqd_star.hypersamples(sample_ind).hyperparameters(end) = ...
%                         qd_gp_mean(star_ind);
%                 end
% 
%                 gpqd_star = train_gp('sqdexp', 'constant', gpqd_star, ...
%                     samples_i, qd_i(:,star_ind), opt);
%                 [best_hypersample, best_hypersample_struct] = ...
%                     disp_hyperparams(gpqd_star);
% 
%                 qd_gp_mean(star_ind) = best_hypersample_struct.mean;
%             end
%             %qd_gp_mean = mean(qd_i,1);
% 
%             qd_gp.quad_mean = qd_gp_mean;
% 
%             parfor star_ind = 1:num_star
% 
%                 gpqdd_star = gpqdd;
%                 for sample_ind = 1:numel(gpqdd_star.hypersamples)
%                     gpqdd_star.hypersamples(sample_ind).hyperparameters(end) = ...
%                         qdd_gp_mean(star_ind);
%                 end
% 
%                 gpqdd_star = train_gp('sqdexp', 'constant', gpqdd_star, ...
%                     samples_i, qdd_i(:,star_ind), opt);
%                 [best_hypersample, best_hypersample_struct] = ...
%                     disp_hyperparams(gpqdd_star);
% 
%                 qdd_gp_mean(star_ind) = best_hypersample_struct.mean;
%             end
%             %qdd_gp_mean = mean(qdd_i,1);
%             qdd_gp_mean = max(qdd_gp_mean, qd_gp_mean.^2);

        
%        qdd_gp.quad_mean = qdd_gp_mean;


        [BQR_mean(:,i), BQR_sd(:,i), BQ_mean(:,i), BQ_sd(:,i)] = ...
            predict(sample_struct, prior, r_gp, qd_gp, qdd_gp);
        
        [BMC_mean(:,i), BMC_sd(:,i)] = predict_BMC(sample_struct, prior, r_gp, qd_gp);

        [MC_mean(:,i), MC_sd(:,i)] = predict_MC(sample_struct, prior);
        
        [ML_mean(:,i), ML_sd(:,i)] = predict_ML(sample_struct, prior);

        perf_BQR(i) = sum(lognormpdf(BQR_mean(:,i),y_star,BQR_sd(:,i)));
        perf_BQ(i) = sum(lognormpdf(BQ_mean(:,i),y_star,BQ_sd(:,i)));
        perf_BMC(i) = sum(lognormpdf(BMC_mean(:,i),y_star,BMC_sd(:,i)));
        perf_MC(i) = sum(lognormpdf(MC_mean(:,i),y_star,MC_sd(:,i)));
        perf_ML(i) = sum(lognormpdf(ML_mean(:,i),y_star,ML_sd(:,i)));
        fprintf('Sample %u\n performance\n BQR:\t%g\n BQ:\t%g\n BMC:\t%g\n MC:\t%g\n ML:\t%g\n',...
            i,perf_BQR(i),perf_BQ(i),perf_BMC(i),perf_MC(i),perf_ML(i));


    %save test_bqr_on_flux_prediction
end
%end

close all
err_BQR = sqrt(mean((bsxfun(@minus, BQR_mean(:,1:i), y_star).^2)));
err_BQ = sqrt(mean((bsxfun(@minus, BQ_mean(:,1:i), y_star).^2)));
err_BMC = sqrt(mean((bsxfun(@minus, BMC_mean(:,1:i), y_star).^2)));
err_MC = sqrt(mean((bsxfun(@minus, MC_mean(:,1:i), y_star).^2)));
err_ML = sqrt(mean((bsxfun(@minus, ML_mean(:,1:i), y_star).^2)));

semilogy(err_BQR, '.k')
hold on
semilogy(err_BQ, '+k')
semilogy(err_BMC, '.r')
semilogy(err_MC, '.m')
semilogy(err_ML, '.b')

load test_bqr_on_flux_prediction

BMC_errs = ~isreal(BMC_sd);
BMC_sd(BMC_errs) = ML_sd(BMC_errs);

BMC_errs = abs(BMC_mean) > 0.5e4;
BMC_mean(BMC_errs) = ML_mean(BMC_errs);
BMC_sd(BMC_errs) = ML_sd(BMC_errs);

BQR_errs = abs(BQR_mean) > 0.5e4;
BQR_mean(BQR_errs) = ML_mean(BQR_errs);
BQR_sd(BQR_errs) = ML_sd(BQR_errs);


BQ_errs = abs(BQ_mean) > 0.5e4;
BQ_mean(BQ_errs) = ML_mean(BQ_errs);
BQ_sd(BQ_errs) = ML_sd(BQ_errs);


meana = @(x) mean(x(:));
rmsne = @(mn,sd) meana(-log(sqrt(2*pi)*sd))-0.5*meana((bsxfun(@minus, mn, y_star)./sd).^2);

     perf_BQR = rmsne(BQR_mean(:,50:i),BQR_sd(:,50:i));
        perf_BQ = rmsne(BQ_mean(:,50:i),BQ_sd(:,50:i));
        perf_BMC = rmsne(BMC_mean(:,50:i),BMC_sd(:,50:i));
        perf_MC = rmsne(MC_mean(:,50:i),MC_sd(:,50:i));
        perf_ML = rmsne(ML_mean(:,50:i),ML_sd(:,50:i));

fprintf('Sample %u\n performance\n BQR:\t%g\n BQ:\t%g\n BMC:\t%g\n MC:\t%g\n ML:\t%g\n',...
            i,perf_BQR,perf_BQ,perf_BMC,perf_MC,perf_ML);

for j = 1:i
     perf_BQR(j) = rmsne(BQR_mean(:,j),BQR_sd(:,j));
     
        perf_BQ(j) = rmsne(BQ_mean(:,j),BQ_sd(:,j));
        perf_BMC(j) = rmsne(BMC_mean(:,j),BMC_sd(:,j));
        perf_MC(j) = rmsne(MC_mean(:,j),MC_sd(:,j));
        perf_ML(j) = rmsne(ML_mean(:,j),ML_sd(:,j));
end

close all
fh = figure;
set(gca, 'FontSize', 24);
set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(fh, 'color', 'white'); 

pMC = semilogx(perf_MC, '-m','LineWidth',1)
hold on
pBMC = semilogx(1:i,perf_BMC, '.b','MarkerSize',3)
pML = semilogx(perf_ML, 'xr','MarkerSize',3)
pBQR = semilogx(perf_BQR, '.k')


axis tight
ylim([-30,-5])

set(gca, 'YGrid', 'off','YTick',[-30 -20 -10]);

set(fh, 'units', 'centimeters');
pos = get(fh, 'position'); 
set(fh, 'position', [pos(1:2), 9, 4]); 


xlab = xlabel('\# samples');
xlabpos = get(xlab,'Position');
xlabpos(1) = xlabpos(1) + 15;
xlabpos(2) = xlabpos(2) + 6;
set(xlab,'Position',xlabpos);
ylab = ylabel('\acro{ll}','Rotation',0)
ylabpos = get(ylab,'Position');
%set(ylab,'Rotation',0);

hleg = mf_legend([pML,pMC,pBMC,pBQR],{'\acro{ml}','\acro{mc}','\acro{nbq}','\acro{bqr}'}, ...
    'NorthWest',3);
legend boxon
set(hleg,'XColor',[1 1 1],'YColor',[1 1 1])
%set(hleg,'Orientation','Horizontal')

matlabfrag('~/Documents/SBQ/flux_perf')


