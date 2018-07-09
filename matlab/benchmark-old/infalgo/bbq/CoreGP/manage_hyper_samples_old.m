function covvy=manage_hyper_samples(covvy,hyperrho,invKL)
% Drop one hypersample, add another

% q*rho is mu in my notes, got to be usable somehow

h2sample_ind=covvy.ML_hyper2sample_ind;

lowr.UT=true;
lowr.TRANSA=true;

samples=cat(1,covvy.hypersamples.hyperparameters);
num_samples=size(samples,1);
candidates=cat(1,covvy.candidates.hyperparameters); 
num_candidates = size(candidates,1);

Nhyperparams=numel(covvy.hyperparams);
hps=1:Nhyperparams;

rearrange=covvy.rearrange;

inputscales=exp(covvy.hyper2samples(h2sample_ind).hyper2parameters);
priorMeans=[covvy.hyperparams.priorMean];
priorSDs=[covvy.hyperparams.priorSD];


toMove = get_hyper_samples_to_move(covvy)

cholK=covvy.hyper2samples(h2sample_ind).cholK;
N=covvy.hyper2samples(h2sample_ind).N;

% downdate
cholK = downdatechol(cholK,toMove);
N(toMove,:) = nan;
Nold=N;
Nold(toMove,:) = [];
% invRN = inv(chol(K)')*N
invRN = linsolve(cholK,Nold,lowr);

% As initial guesses for the minimum of expected_uncertainty, we try both
% making one step of gradient ascent from every hypersample as well as our
% candidates (points that are far away from observations).
num_ascent_points = min(num_samples,8); % Parameter ahoy
step_size=0.1;

num_starting_points = num_candidates+num_ascent_points; % Parameter ahoy

% starting_points consists of gradient ascent points followed by
% candidates. The gradient ascent allows us to avoid the conditioning
% errors we would see in expected_uncertainty if our trial hypersample were
% too close to an existing hypersample.
starting_points = determine_starting_points(covvy, h2sample_ind, num_ascent_points,step_size);

opts.MaxFunEvals=20; % Parameter ahoy
opts.Display='off'; 
opts.LargeScale='off';

added_hypersamples=nan(num_starting_points,Nhyperparams);
uncertainty=nan(num_starting_points,1);
for start=1:num_ascent_points
    %tic;
    [added_hypersamples(start,:),uncertainty(start)]=fminunc(@(trial_hypersample) ...
        expected_uncertainty(trial_hypersample,...
            cholK,N,invRN,toMove,invKL,hyperrho,hps,samples,...
            candidates,inputscales,priorMeans,priorSDs,rearrange),...
            starting_points(start,:),opts);
    %toc;
    % With 20 MaxFunEvals, ~2secs per fminunc call
end

%opts.MaxFunEvals=20; % Parameter ahoy

for start = num_ascent_points + (1:num_candidates)
    %tic;
		added_hypersamples(start,:) = starting_points(start,:);
    uncertainty(start) = expected_uncertainty(starting_points(start,:),...
        cholK,N,invRN,toMove,invKL,hyperrho,hps,samples,...
            candidates,inputscales,priorMeans,priorSDs,rearrange);
%     [added_hypersamples(start,:),uncertainty(start)]=fminunc(@(trial_hypersample) ...
%         expected_uncertainty(trial_hypersample,...
%             cholK,N,invRN,toMove,invKL,hyperrho,hps,samples,...
%             candidates,inputscales,priorMeans,priorSDs,rearrange),...
%             starting_points(start,:),opts);
end

% points=allcombs({0,log(0.2),linspace(-0.1,0.1,500)',0.1});
% 
% unc=nan(500,1);
% for i=1:500
%     unc(i)=expected_uncertainty(points(i,:),...
%         cholK,N,invRN,toMove,invKL,hyperrho,hps,samples,...
%             candidates,inputscales,priorMeans,priorSDs,rearrange);
% end
% hold on
% plot(exp(points(:,3)),unc,'+')
% 
% A=cat(1,covvy.hypersamples.hyperparameters);B=A([2,3,11,14],3);points=allcombs({0,log(0.2),B,0.1});
% unc=nan(4,1);
% for i=1:4
%     unc(i)=expected_uncertainty(points(i,:),...
%         cholK,N,invRN,toMove,invKL,hyperrho,hps,samples,...
%             candidates,inputscales,priorMeans,priorSDs,rearrange);
% end
% 
% plot(exp(points(:,3)),unc,'r+')

[min_uncertainty,min_ind]=min(uncertainty);
added_hypersample=added_hypersamples(min_ind,:);

covvy.hypersamples(toMove).hyperparameters=added_hypersample;
covvy.lastHyperSampleMoved=toMove;

% samples2=cat(1,covvy.hypersamples.hyperparameters);
% if all(abs(samples2(:,3)>1))
%     1
% end

function sigma=expected_uncertainty(trial_hypersample,cholK,N,invRN,add_posn,invKL,hyperrho,hps,samples,candidates,inputscales,priorMeans,priorSDs,rearrange)

num_samples=size(samples,1);
Nhyperparams=length(hps);

samples_wtrial=[samples(1:add_posn-1,:);trial_hypersample;samples(add_posn+1:end,:)];

% Update K and N given this new trial hypersample

Kvec=ones(1,size(samples_wtrial,1));
Nvec=ones(1,size(N,2));
ind=0;
for hyperparam=hps
    ind=ind+1;
    
    width=inputscales(hyperparam);
    priorSD=priorSDs(hyperparam);
    priorMean=priorMeans(hyperparam);
    
    candidates_hp=candidates(:,hyperparam);
    samples_hp=samples(:,hyperparam);
    trial_hypersample_hp=trial_hypersample(hyperparam);
    samples_wtrial_hp=samples_wtrial(:,hyperparam);
    
    K_hp=matrify(@(x,y) normpdf(x,y,width),...
                                trial_hypersample_hp,samples_wtrial_hp);
    Kvec=Kvec.*K_hp;

    
    % in the following, the postscript _A refers to a term over samples and
    % samples, _B to a term over samples and gradient samples and _C to a
    % term over samples and candidates.
    
    % No need to add trial_hypersample to the columns of N, which are
    % purely over samples w derivs and candidates
           
    inds=(Nhyperparams+1-ind)*num_samples+(1:num_samples);

    PrecX=(priorSD^2+width^2-priorSD^4/(priorSD^2+width^2))^(-1);
    PrecY=(priorSD^2-(priorSD^2+width^2)^2/(priorSD^2))^(-1);
    % Nfn2=@(x,y) mvnpdf([x;y],[SamplesMean(d);SamplesMean(d)],...
    %         [SamplesSD(d)^2+width^2,SamplesSD(d)^2;SamplesSD(d)^2,SamplesSD(d)^2+width^2]);
    Nfn=@(x,y) (4*pi^2*(priorSD^2+width^2)/PrecX)^(-0.5)*...
        exp(-0.5*PrecX*((x-priorMean).^2+(y-priorMean).^2)-...
        PrecY.*(x-priorMean).*(y-priorMean));
    N_hp_A=...%diag(normpdf(IndepSamples,SamplesMean(d),SamplesSD(d)));
    matrify(Nfn,trial_hypersample_hp,samples_hp);
    N_hp_C=matrify(Nfn,trial_hypersample_hp,candidates_hp);

    N_hp_B=-width^-2*(samples_hp'-priorMean-...
        matrify(@(x,y) (PrecX+PrecY)*priorSD^2*((x-priorMean)+(y-priorMean)),trial_hypersample_hp,samples_hp));
    % NB: (PrecX+PrecY)*priorSD^2 == priorSD^2/(width^2+2*priorSD^2)

    N_hp=[repmat(N_hp_A,1,Nhyperparams+1),N_hp_C];
    N_hp(:,inds)=N_hp_A.*N_hp_B;
    
    Nvec=Nvec.*N_hp;
end

Knew=nan(size(cholK)+1);
Knew(add_posn,:)=Kvec;
Knew(:,add_posn)=Kvec';

Nnew=N;
Nnew(add_posn,:)=Nvec(:,rearrange);

cholKnew=updatechol(Knew,cholK,add_posn);
if cond(cholKnew'*cholKnew)>10^5
    1;
end
invRNnew=updatedatahalf(cholKnew,Nnew,invRN,cholK,add_posn);

% (NhypersamplesxNtrial_funs)
%  =(NhypersamplesxNhypersamples_wderivs_wcandidates)*(Nhypersamples_wderivs_wcandidatesxNtrial_funs)

% This is a potential bottleneck - can be made faster by reducing number of
% trial fns
d=invRNnew*invKL;

sigma=-ones(1,num_samples)*d.^2*hyperrho';

%sigma=-sum(d.^2,1)*hyperrho';



