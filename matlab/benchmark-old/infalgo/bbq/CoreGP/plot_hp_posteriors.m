function postmean = plot_hp_posteriors(covvy)
%  will produce a separate plot showing the posterior distribution for each
%  hyperparameter, over the [pmean-3*pSD,pmean+3*pSD] range (pmean = prior mean, pSD = prior SD)

Nhps = numel(covvy.hyperparams);
NSamples = 300;
for hp=1:Nhps
    figure;
    locations = nan(NSamples,Nhps);
    pmean = covvy.hyperparams(hp).priorMean;
    pSD = covvy.hyperparams(hp).priorSD;
    locationvec = linspace(pmean-3*pSD,pmean+3*pSD,NSamples)';
    locations(:,hp) = locationvec;
    [postmean,posteriors,exp_postmean,exp_posteriors]=posterior_hp(covvy,locations);
    post = plot(locationvec,posteriors,'r-','LineWidth',2);
    xlabel(covvy.hyperparams(hp).name);
    ylabel('Posterior density');
    title(['Posterior density for hyperparam ',num2str(hp)]);
    
    mn=line(([postmean(hp) postmean(hp)]),[min(posteriors) max(posteriors)],'LineWidth',1,'Color','r','LineStyle','--');
    legend([post,mn],'Posterior','Posterior Mean','Location','EastOutside')

    exp_post = plot(exp(locationvec),exp_posteriors,'r-','LineWidth',2);
    xlabel(covvy.hyperparams(hp).name);
    ylabel('Posterior density');
    title(['Posterior density for hyperparam ',num2str(hp)]);
    
    exp_mn=line(([exp_postmean(hp) exp_postmean(hp)]),[min(exp_posteriors) max(exp_posteriors)],'LineWidth',1,'Color','r','LineStyle','--');
    legend([exp_post,exp_mn],'Posterior','Posterior Mean','Location','EastOutside')
    
end
