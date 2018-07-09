clear
%fn=@sqrt;
%invfn=@(x) x.^2;
Ndims=4;
bnd=1;
%XData=(-5:1:9)';
XData=[kron2d(allcombs(repmat((-bnd:bnd)',1,Ndims)),ones(Ndims+1,1)),repmat([zeros(1,Ndims);eye(Ndims)],(2*bnd+1)^Ndims,1)];
NData=length(XData);
YData=normpdf(XData(:,1),0,2)+normpdf(XData(:,2),3,5)+normpdf(XData(:,3),-2,2.5);
XStar=allcombs({repmat((-1:1)',1,Ndims),ones(1,Ndims)});

covvy=struct('covfn',@(hp,varargin) ndimsqdexp_isotropic_cov_fn_withderivs(hp,Ndims,varargin{:}));

mu=0.01;
L=1;
T=0.66;

covvy.hyperparams(1)=struct('name','logInputScale1','priorMean',log(T),'priorSD',0.2,'NSamples',1,'type','real');
covvy.hyperparams(2)=struct('name','logOutputScale','priorMean',log(L),'priorSD',0.7,'NSamples',1,'type','real');%30
covvy.hyperparams(3)=struct('name','mean','priorMean',mu,'priorSD',1,'NSamples',1,'type','real');
covvy.hyperparams(4)=struct('name','logNoiseSD','priorMean',-inf,'priorSD',0,'NSamples',1,'type','real');

covvy=hyperparams(covvy);
covvy=bmcparams(covvy);
covvy=gpparams(XData,YData,covvy,'overwrite');

[m,C]=gpmeancov(XStar,XData,covvy);

clf
hold on
% plot(XData,YData,'+k')
% plot(XStar,m,'b')

%plot(XStar,m,'r')
%plot(XStar,exp(mu)*(1+(m-mu)+0.5*(m-mu).^2+1/6*(m-mu).^3+1/24*(m-mu).^4+1/120*(m-mu).^5+1/720*(m-mu).^6),'b')
%plot_regression(XData,YData,XStar,m,sqrt(diag(C)))



% rho=covvy.hypersamples(1).datatwothirds*L^2;
% f=nan*XStar;
% for i=1:length(XStar)
%     f(i)=exp(mu)*prod(1+(exp(rho)-1).*exp(-1/2*((XStar(i)-XData)./T).^2.*(1+0.35*abs(rho))));
% end
% plot(XStar,f,'b')
% 
% sum(exp(m))
% sum(f)