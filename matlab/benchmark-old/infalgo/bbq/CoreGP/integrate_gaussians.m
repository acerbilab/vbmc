function [estimated_value,real_value,covvy,monitor] = integrate_gaussians(method,covvy,likelihood_params,q_params,num_steps)

num_hps = numel(covvy.hyperparams);

if ~isfield(covvy,'active_hp_inds')
    active=[];
    for hyperparam = 1:num_hps
        if ~strcmpi(covvy.hyperparams(hyperparam).type,'inactive')
            active=[active,hyperparam];
        else
            covvy.hyperparams(hyperparam).NSamples=1;
        end
    end
    covvy.active_hp_inds=active;
end

likelihood_fn = @(covvy,samples) gaussian_likelihood_fn(covvy,samples,likelihood_params);
q_fn  = @(covvy,samples) gaussian_q_fn(covvy,samples,q_params);

load candidate_combs_cell2;

if covvy.plots
    covvy2=covvy;
    for hyperparam=1:num_hps
        covvy2.hyperparams(hyperparam).NSamples=100;
        covvy2.hyperparams(hyperparam).samples=[];
    end

    covvy2=hyperparams(covvy2);
    %covvy2=gpparams(XData(:,2),YData(:),covvy2,'overwrite',[],1:1000);
    covvy2=likelihood_fn(covvy2,1:(100^num_hps));

    covvy.real_logLs = [covvy2.hypersamples.logL]';
    covvy.real_hps = cat(1,covvy2.hypersamples.hyperparameters);
end

switch method
    case 'AHS'
        [estimated_value,covvy,monitor]=integrate_ahs(covvy,q_fn,likelihood_fn,candidate_combs_cell,num_steps);
    case 'HMC'
        [estimated_value,covvy]=integrate_HMC(covvy,q_fn,likelihood_fn,candidate_combs_cell,num_steps);
    case 'ML'
        [estimated_value,covvy]=integrate_ML(covvy,q_fn,likelihood_fn,candidate_combs_cell,num_steps);
end

% determine real_value

q_samples = q_params.inputs;
likelihood_samples = likelihood_params.inputs;
num_q_samples = length(q_samples);
num_likelihood_samples = length(likelihood_samples);

inputscales_Q = q_params.input_scales;
inputscales_L = likelihood_params.input_scales;



K_Q = ones(num_q_samples);
K_L = ones(num_likelihood_samples);
N = ones(num_q_samples,num_likelihood_samples);
n = ones(num_likelihood_samples,1);

for hyperparam=1:num_hps
    width_L=inputscales_L(hyperparam);
    width_Q=inputscales_Q(hyperparam);
    q_samples_hp=q_samples(:,hyperparam);
    likelihood_samples_hp=likelihood_samples(:,hyperparam);

    priorMean=covvy.hyperparams(hyperparam).priorMean;
    priorSD=covvy.hyperparams(hyperparam).priorSD;
    
    K_Q_hp=matrify(@(x,y) normpdf(x,y,width_Q),...
                    q_samples_hp,q_samples_hp);
    K_Q=K_Q.*K_Q_hp;
    
    K_L_hp=matrify(@(x,y) normpdf(x,y,width_L),...
                likelihood_samples_hp,likelihood_samples_hp);
    K_L=K_L.*K_L_hp;

    n_hp=normpdf(likelihood_samples_hp,priorMean,sqrt(width_L^2+priorSD^2));
    n=n.*n_hp;

    determ=priorSD^2*(width_L^2+width_Q^2)+width_L^2*width_Q^2;
    PrecX_L=(priorSD^2+width_L^2)/determ;
    PrecX_Q=(priorSD^2+width_Q^2)/determ;
    PrecY=-priorSD^2/determ;
    % Nfn2=@(x,y) mvnpdf([x;y],[SamplesMean(d);SamplesMean(d)],...
    %         [SamplesSD(d)^2+width^2,SamplesSD(d)^2;SamplesSD(d)^2,SamplesSD(d)^2+width^2]);
    Nfn=@(x,y) (4*pi^2*determ)^(-0.5)*...
        exp(-0.5*PrecX_L*(x-priorMean).^2-0.5*PrecX_Q*(y-priorMean).^2-...
        PrecY.*(x-priorMean).*(y-priorMean));
    N_hp=...%diag(normpdf(IndepSamples,SamplesMean(d),SamplesSD(d)));
    matrify(Nfn,q_samples_hp,likelihood_samples_hp);

    N=N.*N_hp;
end

real_value = (q_params.outputs'*inv(K_Q)*N*inv(K_L)*likelihood_params.outputs)/(n'*inv(K_L)*likelihood_params.outputs);

function covvy = gaussian_likelihood_fn(covvy,samples,params)
if isempty(samples)
    return
end
inputs = params.inputs;
outputs = params.outputs;
input_scales = params.input_scales;
%output_scales = params.output_scales;
active_hp_inds = covvy.active_hp_inds;
num_dims = numel(covvy.hyperparams);
num_samples = length(samples);

%num_hypersamples = numel(covvy.hypersamples);
phis = cat(1,covvy.hypersamples(samples).hyperparameters);
phis = phis(:,active_hp_inds);
[L,gL] = simple_gpmean(phis,inputs,outputs,'gaussian',{input_scales,1});

gLmat = cat(2,gL{:});
glogLmat = zeros(num_samples,num_dims);
glogLmat(:,active_hp_inds) = gLmat./repmat(L,1,size(gLmat,2));
glogLmat(isnan(glogLmat)) = 0;

for ind=1:num_samples
    sample=samples(ind);
    covvy.hypersamples(sample).logL = max(log(L(ind)),-700);
    covvy.hypersamples(sample).glogL = mat2cell(glogLmat(ind,:)',num_dims,1);
end

function qs = gaussian_q_fn(covvy,samples,params)
if isempty(samples)
    qs=[];
    return
end

inputs = params.inputs;
outputs = params.outputs;
input_scales = params.input_scales;
%output_scales = params.output_scales;
active_hp_inds = covvy.active_hp_inds;

num_hypersamples = numel(covvy.hypersamples);
phis = cat(1,covvy.hypersamples(samples).hyperparameters);
phis = phis(:,active_hp_inds);
qs = nan(num_hypersamples,1);
qs(samples) = max(simple_gpmean(phis,inputs,outputs,'gaussian',{input_scales,1}),exp(-700));