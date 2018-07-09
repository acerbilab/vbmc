clear;

%


prob_bqr_incr_samples;

%BQR = [];BQ=[];BMC=[];MC=[];
load test_bqr_incr_samples_hmc
f = isnan(prod(MC,2));
BQR(f,:) = [];BQ(f,:)=[];BMC(f,:)=[];MC(f,:)=[];

neg_log_rp = @(x) - log(r_fn(x)) - log(p_fn(x));
d_neg_log_rp = @(x) - d_r_fn(x)/r_fn(x) - d_p_fn(x)/p_fn(x);

hmc2('state', 0);
hmc_options = struct('nsamples',max_num_samples*3,...
        'nomit',0,'display',0,'stepadj',prior.sds);
hmc_options = hmc2_opt(hmc_options);

%     samples = ...
%         hmc2(neg_log_rp, 0, hmc_options, d_neg_log_rp);
%     clf;hold on;
% ezplot(@(x) exp(-neg_log_rp(x)))
% ff=[];qq=[];for i=1:max_num_samples;ff(i)=exp(-neg_log_rp(samples(i)));qq(i)=q_fn(samples(i));end;
% plot(samples,ff,'b.','MarkerSize',5)
% plot(samples,qq,'r.','MarkerSize',5)
% axis tight
% 
% figure
% histfit(samples,50)
% h = get(gca,'Children');
% set(h(2),'FaceColor',[.8 .8 1])


q = [];
r = [];
for trial = (size(BQR,1)+1):max_trials
    fprintf('trial = %u\n', trial);
    
    BQR = [BQR;nan(1, max_num_samples)];
    BQ = [BQ;nan(1, max_num_samples)];
    BMC = [BMC;nan(1, max_num_samples)];
    MC = [MC;nan(1, max_num_samples)];
    
    samples = ...
        hmc2(neg_log_rp, prior.means, hmc_options, d_neg_log_rp);
    [unique_samples, inds] = unique(samples);
    samples = samples(sort(inds));
    samples = samples(1:max_num_samples);

    q= [];
    r= [];
    gpq = [];
    gpr = [];
    for num_sample = 1:max_num_samples;
        fprintf('%g,',num_sample);
           
        % structs for q and r no longer reqd
        q = [q;q_fn(samples(num_sample,:))];
        r = [r;r_fn(samples(num_sample,:))];
        
        sample_struct.samples = samples(1:num_sample,:);
        sample_struct.log_r = log(r);
        sample_struct.q = q;
        
%         [r_noise_sd, r_input_scales, r_output_scale] = ...
%             hp_heuristics(samples, r, 100);

  
        gpq = train_gp('sqdexp', 'constant', gpq, ...
            samples(1:num_sample,:), q, opt);
        [best_hypersample, best_hypersample_struct] = disp_hyperparams(gpq);
        
        q_gp.quad_output_scale = best_hypersample_struct.output_scale;
        q_gp.quad_input_scales = best_hypersample_struct.input_scales;
        q_gp.quad_noise_sd = best_hypersample_struct.noise_sd;
        q_gp.quad_mean = best_hypersample_struct.mean;
        
        gpr = train_gp('sqdexp', 'constant', gpr, ...
            samples(1:num_sample,:), r, opt);
        [best_hypersample, best_hypersample_struct] = disp_hyperparams(gpr);
        
        r_gp.quad_output_scale = best_hypersample_struct.output_scale;
        r_gp.quad_input_scales = best_hypersample_struct.input_scales;
        r_gp.quad_noise_sd = best_hypersample_struct.noise_sd;
        r_gp.quad_mean = best_hypersample_struct.mean;
   
        [BQR(trial,num_sample), dummy, BQ(trial,num_sample)] = ...
            predict(sample_struct, prior, r_gp, q_gp);
                
        BMC(trial,num_sample) = predict_BMC(sample_struct, prior, r_gp, q_gp);
        
        MC(trial,num_sample) = predict_MC(sample_struct, prior);
        
    end
    save test_bqr_incr_samples_hmc

            i = trial;

        perf_BQR = sqrt(((BQR(i,end) - exact).^2));
        perf_BQ = sqrt((abs(BQ(i,end) - exact).^2));
        perf_BMC = sqrt((abs(BMC(i,end) - exact).^2));
        perf_MC = sqrt((abs(MC(i,end) - exact).^2));
        std_BQR = sqrt(std((BQR(i,:) - exact).^2));
        std_BQ = sqrt(std((BQ(i,:) - exact).^2));
        std_BMC = sqrt(std((BMC(i,:) - exact).^2));
        std_MC = sqrt(std((MC(i,:) - exact).^2));
        fprintf('Trial %u\n performance\n BQR:\t%g\t+/-%g\n BQ:\t%g\t+/-%g\n BMC:\t%g\t+/-%g\n MC:\t%g\t+/-%g\n',...
            i,perf_BQR,std_BQR,perf_BQ,std_BQ,perf_BMC,std_BMC,perf_MC,std_MC);

%         figure;hold on;
%         plot(exact+0*MC(i,:),'k')
%         plot(BQR(i,:),'r')
%         plot(BMC(i,:),'b')
%         plot(MC(i,:),'m')

    %end
            
%         f = all(~isnan(MC'))';
%     
%     
%             figure;hold on;
%         plot(exact+0*MC(1,:),'k')
%         plot(mean(BQR(f,:)),'r')
%         plot(mean(BMC(f,:)),'b')
%         plot(mean(MC(f,:)),'m')
%     
    
    fprintf('\n');
end


load test_bqr_incr_samples_hmc
        perf_BQR = sqrt(mean(abs(BQR(:,end) - exact).^2));
        perf_BQ = sqrt(mean(abs(BQ(:,end) - exact).^2));
        perf_BMC = sqrt(mean(abs(BMC(:,end) - exact).^2));
        perf_MC = sqrt(mean(abs(MC(:,end) - exact).^2));
%         std_BQR = sqrt(mean(std((BQR(:,end) - exact).^2)));
%         std_BQ = sqrt(mean(std((BQ(:,end) - exact).^2)));
%         std_BMC = sqrt(mean(std((BMC(:,end) - exact).^2)));
%         std_MC = sqrt(mean(std((MC(:,end) - exact).^2)));
        fprintf('Trial %u\n performance\n BQR:\t%g\n BQ:\t%g\n BMC:\t%g\n MC:\t%g\n',...
            i,perf_BQR,perf_BQ,perf_BMC,perf_MC);
        
        

% colours = colormap('hsv'); 
% inds = ceil(linspace(3*length(colours)/4,length(colours),4));

load test_bqr_incr_samples_hmc
clf
fh = figure;
set(gca, 'FontSize', 24);
set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(fh, 'color', 'white'); 

loglog(sqrt(mean(bsxfun(@minus,MC,exact).^2)),'-m','LineWidth',1);
hold on;
loglog(sqrt(mean(bsxfun(@minus,BMC,exact).^2)),'.b','MarkerSize',3);
loglog(sqrt(mean(bsxfun(@minus,BQ,exact).^2)),'xr','MarkerSize',2);
loglog(sqrt(mean(bsxfun(@minus,BQR,exact).^2)),'.k');

axis([0 200 0 0.2])
set(gca, 'YGrid', 'off','YTick',[0.01,0.1]);

set(fh, 'units', 'centimeters');
pos = get(fh, 'position'); 
set(fh, 'position', [pos(1:2), 9, 4]); 

legend('\acro{mc}','\acro{nbq}','\acro{bqz}','\acro{bqr}', ...
    'Location','SouthWest')

legend boxoff
title '\acro{hmc} sampling'
xlab = xlabel('\# samples');
xlabpos = get(xlab,'Position');
xlabpos(1) = xlabpos(1)+17;
xlabpos(2) = xlabpos(2)+0.0005;
set(xlab,'Position',xlabpos);
ylabel('\acro{rmse}','Rotation',0)





matlabfrag('~/Documents/SBQ/hmc_sampled_simple')


