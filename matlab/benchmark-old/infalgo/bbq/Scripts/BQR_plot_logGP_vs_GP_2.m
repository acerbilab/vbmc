clear
width = 9.5;
height = 3;


XData = [1,4,13,8,16]';
%q_data = rand(5,1);%[0.1,3,0.1,2,9]';
%q_data = [0.1,3,0.001,2,100]'

q_data =     10000*[0.1289
    0.4607
    0.9816
    0.0564
    0.8555];

% XData = 1;
% q_data = 1;

mu_t_r = 100*max(q_data);
tilde = @(q) log(q/mu_t_r + 1);

q = linspace(0,2*max(q_data),1000);
figure(3)
plot(q, tilde(q), '.')


XStar = linspace(min(XData)-15,max(XData)+15,1000)';


covvy=struct('covfn',@(varargin) versatile_cov_fn('sqdexp',varargin{:}));

% set the priors for our hyperparameters. NSamples is the number of samples
% taken in the initial grid for that hyperparameter. After the action of
% ML/HMC/AHS, of course, those samples will almost certainly be moved into
% non-grid locations.
covvy.hyperparams(1)=struct('name','mean',...
    'priorMean',0,'priorSD',0.0001,'NSamples',1,'type','inactive');
covvy.hyperparams(numel(covvy.hyperparams)+1)=struct('name','logNoiseSD',...
    'priorMean',-100,'priorSD',0.5,'NSamples',1,'type','inactive');
covvy.hyperparams(numel(covvy.hyperparams)+1)=struct('name','logInputScale1',...
    'priorMean',log(0.5),'priorSD',0.6,'NSamples',100,'type','real');
covvy.hyperparams(numel(covvy.hyperparams)+1)=struct('name','logOutputScale',...
    'priorMean',log(1),'priorSD',2,'NSamples',1,'type','inactive');

% Number of iterations to allow the integration algorithm        
num_steps=100;

% perform the predictions using either predict_ML_old, predict_HMC or
% predict_ahs
[mean_r,sd_r,covvy2,closestInd]=predict_ML_old(XStar,XData,q_data,covvy,num_steps);
covvy2.hypersamples(closestInd).hyperparameters

hps = covvy2.hypersamples(closestInd).hyperparameters;
q_input_scale = exp(hps(3));


%[sd,obs,mn] = laplot_nice('Regression1',XData,q_data,XStar,mean_r,sd_r,'$x$','$y$',[],[1 0 0],'wide');
%axis([0 11 2 7])
%set(gca,'XTickLabel',{'-pi','-pi/2','0','pi/2','pi'})
%legend([sd,mn,obs],'$\pm$ $$ 1SD','Mean','Observations','Location','NorthEast')




figure(1)
clf
scrsz = get(0,'ScreenSize');
% 
set(gcf,'units','centimeters')
set(gcf,'Position',[1 1 width height])


hold on

ylabel('$\tilde{r}$','Rotation',0);


covvy.hyperparams(4)=struct('name','logOutputScale',...
    'priorMean',log(0.005),'priorSD',2,'NSamples',1,'type','real');

[mean_t_r,sd_t_r,covvy2,closestInd]=...
    predict_ML_old(XStar,XData,tilde(q_data),covvy,num_steps);
hps = covvy2.hypersamples(closestInd).hyperparameters;

length_scales = exp(hps(3));
t_q_input_scale = length_scales;

% mean_t_r = exp(mean_r2-sd_r2.^4)*Sig;
% TildeYNegSD = exp(mean_r2-sd_r2.^4+sd_r2.^2)*Sig;
% TildeYPosSD = exp(mean_r2-sd_r2.^4-sd_r2.^2)*Sig;


t_mean_r = tilde(mean_r);

axis([min(XStar),max(XStar),-2e-3,11e-3])
%axis([0 15 -5 100])
rmn = plot(XStar,(mean_t_r),'Color',[0 0 0.8]);
amn = plot(XStar,(t_mean_r),'--','Color',[0.8 0 0]);

obs = plot(XData,tilde(q_data),'.k','MarkerSize',12)

mf_legend([rmn,amn,obs],{'$m_{\tr|s}$','$t_r(m_{r|s})$','$\tvr_s$'},'EastOutside',6)


xlab=xlabel('$\phi$');
xlabpos = get(xlab,'Position');
xlabpos(1) = xlabpos(1)-3;
xlabpos(2) = xlabpos(2)+0.004;
set(xlab,'Position',xlabpos);

set(gca, 'TickDir', 'out')
matlabfrag('~/Docs/SBQ/logGP_vs_GP')
fh = figure(2)
clf

hold on




epsil = mean_r .* (mean_t_r-t_mean_r);


xlim([min(XStar),max(XStar)])

set(gcf,'units','centimeters','Position',[1 1 width height])

%obs = plot(XData,0*XData,'.k','MarkerSize',14)

%length_scales = exp(hps(3));
box = [min(XData)-3*length_scales;max(XData)+3*length_scales];
xc = find_farthest(XData, box, 11, length_scales);

x_cs = [xc;XData];
eps_cs = [epsil(find_list(XStar,xc));0*XData];

[EpsMean,EpsSD, covvy3, best_ind]=predict_ML_old(XStar,x_cs,eps_cs,covvy,num_steps);


hps = covvy3.hypersamples(closestInd).hyperparameters;
eps_input_scale = exp(hps(3));

sdp = shaded_sd(XStar,EpsMean,EpsSD);
obs = plot(x_cs,eps_cs,'MarkerSize',12);
mn = plot(XStar,EpsMean);
dif = plot(XStar,epsil,'k','LineWidth', 0.75);

set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(fh, 'color', 'white'); 
set(gca, 'YGrid', 'off');

set(mn, ... 
  'LineStyle', '-', ...
  'LineWidth', 0.75, ...
  'Marker', 'none', ...
  'MarkerSize', 10, ...
  'Color', [0 0 0.8] ...
  );

set(obs, ... 
  'LineStyle', 'none', ...
  'LineWidth', 0.5, ...
  'Marker', '.', ...
  'Color', [0.2 0.2 0.2] ...
  );

%[0.25 0.8 0.2 0.2]

a= mf_legend([dif,obs,mn,sdp],{'$\de$','$\vect{\epsilon}_{rr,c}$','$m_{\de|c}$','$\pm\sqrt{C}_{\de|c}$'},'EastOutside',6);%,'Orientation','Horizontal')

ylabel('$\de$','Rotation',0);
set(gca, 'TickDir', 'out')
ylim([-0.6 1.2])

xlab=xlabel('$\phi$');
xlabpos = get(xlab,'Position');
xlabpos(1) = xlabpos(1)-3;
xlabpos(2) = xlabpos(2) + 0.5;
set(xlab,'Position',xlabpos);

matlabfrag('~/Docs/SBQ/logGP_vs_GP_delta')


%cd ~/Documents/mikesthesis/contents/Marginalising/
%postlaprint(gcf,'logGP_vs_GP','thesis_full')


%x=linspace(0,10,1000);figure;f=1./x.*normpdf(log(x),1,0.1);plot(x,f)