cd ~/Code/gp-code-osborne/
clear
% addpath(genpath('~/Code/gp-code-osborne/'))
% addpath ~/Code/lightspeed
% addpath(genpath('~/Code/Utils/'))
% rmpath ~/Code/CoreGP
% rmpath ~/Code/BQR

matlabpool close
matlabpool open
clear
num_data = 100;
num_star = 100;

opt.print = false;
opt.optim_time = 30;
opt.noiseless = true;
opt.verbose = false;

max_num_samples = 500;


X_data = rand(num_data,5);

y_fn = @(x) 10*sin(pi*x(:,1).*x(:,2)) ...
            + 20*(x(:,3)-0.5).^2 ...
            + 10*x(:,4) ...
            + 5*x(:,5);
y_data = y_fn(X_data) + randn(num_data,1);

X_star = rand(num_star, 5);
y_star = y_fn(X_star);

save prob_bqr_on_GP_prediction;

gp = set_gp('sqdexp','constant', [], X_data, y_data, ...
    1);

% we do not marginalise over priorMean
active_hp_inds = 1:numel(gp.hyperparams);
gp.active_hp_inds = active_hp_inds;


for i = 1:numel(gp.hyperparams)
    gp.hyperparams(i).priorMean = 0;
    gp.hyperparams(i).priorSD = 2;
    % NB: R&Z put prior on noise and signal VARIANCE; we place prior on
    % standard deviation.
end

prior_means = vertcat(gp.hyperparams(active_hp_inds).priorMean);
prior_sds = vertcat(gp.hyperparams(active_hp_inds).priorSD);
prior.means = prior_means;
prior.sds = prior_sds;

p_fn = @(x) mvnpdf(x, prior_means', diag(prior_sds.^2));
r_fn = @(x) exp(log_gp_lik(x, X_data, y_data, gp));

p_r_fn = @(x) p_fn(x) * r_fn(x);
        
    samples = slicesample(prior_means', max_num_samples,...
        'pdf', p_r_fn,'width', prior_sds');
 
load prob_bqr_on_GP_prediction

    for i = 1:max_num_samples
        gp.hypersamples(i).hyperparameters(active_hp_inds) = samples(i,:);
        gp.hypersamples(i).hyperparameters(8) = mean(y_data);
    end
    gp.grad_hyperparams = false;
    gp = revise_gp(X_data, y_data, gp, 'overwrite');
    
    log_r = vertcat(gp.hypersamples.logL);

    mean_y = nan(num_star, max_num_samples);
    var_y = nan(num_star, max_num_samples);
    for hs = 1:max_num_samples
        [mean_y(:, hs), var_y(:, hs)] = ...
            posterior_gp(X_star,gp,hs,'var_not_cov');
    end

    mean_y = mean_y';
    var_y = var_y';

    qd = mean_y;
    qdd = var_y + mean_y.^2;
    

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
        opt.active_hp_inds = 2:10;
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

        README = 'q at max logL value for both BMC and BQR';
        save test_bqr_on_GP_prediction_ML
    end
%end

load test_bqr_on_GP_prediction_ML
meana = @(x) mean(x(:));

perf_BQR = sqrt(meana((bsxfun(@minus, BQR_mean(:,50:i), y_star).^2)));
perf_BQ = sqrt(meana((bsxfun(@minus, BQ_mean(:,50:i), y_star).^2)));
perf_BMC = sqrt(meana((bsxfun(@minus, BMC_mean(:,50:i), y_star).^2)));
perf_MC = sqrt(meana((bsxfun(@minus, MC_mean(:,50:i), y_star).^2)));
perf_ML = sqrt(meana((bsxfun(@minus, ML_mean(:,50:i), y_star).^2)));


fprintf('Sample %u\n performance\n BQR:\t%g\n BQ:\t%g\n BMC:\t%g\n MC:\t%g\n ML:\t%g\n',...
            i,perf_BQR,perf_BQ,perf_BMC,perf_MC,perf_ML);

close all
err_BQR = sqrt(mean((bsxfun(@minus, BQR_mean(:,1:i), y_star).^2)));
err_BQ = sqrt(mean((bsxfun(@minus, BQ_mean(:,1:i), y_star).^2)));
err_BMC = sqrt(mean((bsxfun(@minus, BMC_mean(:,1:i), y_star).^2)));
err_MC = sqrt(mean((bsxfun(@minus, MC_mean(:,1:i), y_star).^2)));
err_ML = sqrt(mean((bsxfun(@minus, ML_mean(:,1:i), y_star).^2)));

fh = figure;
set(gca, 'FontSize', 24);
set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(fh, 'color', 'white'); 

pMC = loglog(err_MC, '-m','LineWidth',1)
hold on
pBMC = loglog(err_BMC, '.b','MarkerSize',3)
pML = loglog(err_ML, 'xr','MarkerSize',3)
pBQR = loglog(err_BQR, '.k')


axis tight
%ylim([-30,-5])

set(gca, 'YGrid', 'off')%,'YTick',[-30 -20 -10]);

set(fh, 'units', 'centimeters');
pos = get(fh, 'position'); 
set(fh, 'position', [pos(1:2), 9, 4]); 


xlab = xlabel('\# samples');
xlabpos = get(xlab,'Position');
xlabpos(1) = xlabpos(1) + 15;
xlabpos(2) = xlabpos(2) + 0.25;
set(xlab,'Position',xlabpos);
ylab = ylabel('\acro{rmse}','Rotation',0)
ylabpos = get(ylab,'Position');
%set(ylab,'Rotation',0);

hleg = mf_legend([pML,pMC,pBMC,pBQR],{'\acro{ml}','\acro{mc}','\acro{nbq}','\acro{bqr}'}, ...
    'SouthWest',1);
%set(hleg,'Orientation','Horizontal')

matlabfrag('~/Documents/SBQ/GP_perf')
