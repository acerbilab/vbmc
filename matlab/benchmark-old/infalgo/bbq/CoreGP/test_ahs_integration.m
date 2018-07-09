% clear
% likelihood_params.inputs = [-0.4 -0.4;0.2 0.2;0.6 1];
% likelihood_params.outputs = [1;1;1];
% likelihood_params.input_scales = [0.1 0.1];
% 
% q_params.inputs = [-0.4 -0.4;0.6 1];
% q_params.outputs = [1;1];
% q_params.input_scales = [0.1 0.1];
% 
% covvy.hyperparams(1)=struct('name','arbitrary',...
%     'priorMean',1, 'priorSD',2, 'NSamples',5, 'type','real');
% %covvy.hyperparams(1).samples=[-5,-4,-3,3,4]';
% covvy.hyperparams(2)=struct('name','arbitrary2',...
%     'priorMean',1, 'priorSD',2, 'NSamples',1, 'type','inactive');
% covvy.plots=false;
% covvy.hyperparams(3)=struct('name','arbitrary2',...
%     'priorMean',1, 'priorSD',2, 'NSamples',5, 'type','real');
% covvy.plots=false;

clear
likelihood_params.inputs = [-0.5;0.5;1];
likelihood_params.outputs = [1;2;4];
likelihood_params.input_scales = [0.2];

q_params.inputs = [-0.5;0.75];
q_params.outputs = [0.6;4];
q_params.input_scales = [0.05];

covvy.hyperparams(1)=struct('name','arbitrary',...
    'priorMean',1, 'priorSD',1, 'NSamples',7, 'type','real');

% covvy.hyperparams(1).samples=[-2,-1,2,3,4]';
% 
% covvy.hyperparams(1).samples=[-2,-0.4,1.4,3,4]';
% covvy.hyper2samples(1).hyper2parameters=log(0.05);
% covvy.hyper2samples(2).hyper2parameters=log(0.3);

covvy.plots=true;


[estimated_value_AHS,real_value,covvy,monitor] = integrate_gaussians('AHS',covvy,likelihood_params,q_params,100);

[estimated_value_ML,real_value,covvy] = integrate_gaussians('ML',covvy,likelihood_params,q_params,100);

[estimated_value_HMC,real_value,covvy] = integrate_gaussians('HMC',covvy,likelihood_params,q_params,100);
save integrate9