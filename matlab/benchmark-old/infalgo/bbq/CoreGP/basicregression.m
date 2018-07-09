
% mn=0;sd=0.3;
% XData=linspacey(mn-2*sd,mn+2*sd,50)';
% NData=length(XData);
% YData=normpdf(XData,mn,sd);
% XStar=(mn-6*sd:0.1:mn+6*sd)';

XData=samples(:,3);%(-1:0.1:1)';
NData=length(XData);
YData=tilda_qs;%ones(NData,1);
XStar=(-6:0.01:12)';





covvy=struct('covfn',@oned_cov_fn);

% mu=0;
% L=1;
% T=sd;

%inputscales=tildaQ_hyperscales;

T=exp(inputscales(3));
L=sqrt(h_tildaQ^2/sqrt(prod(2*pi*inputscales.^2)));
mu=mean_tildaq;
        
covvy.hyperparams(1)=struct('name','logTimeScale','priorMean',log(T),'priorSD',0.2,'NSamples',1,'type','real');
covvy.hyperparams(2)=struct('name','logLengthScale','priorMean',log(L),'priorSD',0.7,'NSamples',1,'type','real');%30
covvy.hyperparams(3)=struct('name','mean','priorMean',mu,'priorSD',1,'NSamples',1,'type','real');
covvy.hyperparams(4)=struct('name','logNoiseSD','priorMean',-16,'priorSD',0.6,'NSamples',1,'type','real');
        
covvy=hyperparams(covvy);
covvy=bmcparams(covvy);
covvy=gpparams(XData,YData,covvy,'overwrite');
rho=weights(covvy);


% K=covvy.covfn(covvy.hypersamples(1).hyperparameters);
% cond(K(XData,XData))

[m,C]=weighted_gpmeancov(rho,XStar,XData,covvy);
sig=sqrt(C);

figure
hold on
fill([XStar;flipud(XStar)],[m+sig;flipud(m-sig)],[0.9,0.9,0.9],'EdgeColor',[0.9,0.9,0.9])
 plot(XData,YData,'+k')
% plot(XStar,m,'b')
plot(XStar,m,'b')