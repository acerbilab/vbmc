function [postmean,posteriors,exp_postmean,exp_posteriors]=posterior_hp_ahs(covvy,locations)
% postmean is the posterior mean for the hyperparameters.
% If locations is included, an n*d matrix where n is the number of
% locations and d is the dimension of the hyperparameter space, the
% posterior distribution is evaluated at those points in hyperparameter
% space. If any columns of locations are nans, it is assumed that the
% relevant hyperparameter is to be marginalised. This allows the posterior
% over individual (or subsets of) hyperparameters to be obtained.
% Assumes distribution over likelihoods is effectively a delta distribution
% around the mean likelihood function, also that our hypersamples are
% distributed according to a grid.

% This function will not work for ahs, just needs to have hypersamples used in the place of covvy. krons replaced
% with .*'s though

full_samples = cat(1,covvy.hypersamples.hyperparameters);
num_hyperparams=numel(covvy.hyperparams);
num_samples=numel(covvy.hypersamples);

want_posteriors=nargin>1 && nargout>1;
if want_posteriors
    num_locations=size(locations,1);
    post_numer=ones(num_locations,1);
    n_locations_samples = ones(num_locations, num_samples);
    exp_locations = ones(num_locations,1);
end


germank = ones(num_hyperparams,num_samples);
exp_germank = ones(num_hyperparams,num_samples);
Ks = ones(num_samples);
ns = ones(1, num_samples);







if isfield(covvy,'hyper2samples') && isfield(covvy,'ML_hyper2sample_ind')
    widths = exp(covvy.hyper2samples(covvy.ML_hyper2sample_ind).hyper2parameters);
else
    widths = nan(1,num_hyperparams);
    for hyperparam=1:num_hyperparams
        samples = covvy.hyperparams(hyperparam).samples;
        widths(hyperparam)=0.38*separation(samples);
    end
end


for hyperparam=1:num_hyperparams
    
    samples = full_samples(:,hyperparam);

    priorMean=covvy.hyperparams(hyperparam).priorMean;
    priorSD=covvy.hyperparams(hyperparam).priorSD;
    if isfield(covvy.hyperparams,'type')
        type  = covvy.hyperparams(hyperparam).type;
    else
        type = 'real';
    end
    if ~all(~isnan([samples;priorMean;priorSD]))
        type = 'inactive';
    end
    
    if strcmp(type,'inactive')
        % This hyperparameter is a dummy - ignore it
        continue
    end
    
    
    width=widths(hyperparam);
    
    if want_posteriors
        locations_hp = locations(:,hyperparam);
        if strcmp(type,'real') && ~any(isnan(locations_hp))
            exp_locations = exp_locations.*exp(locations_hp);
        end
    end
    
    switch type
        case 'real'
            ns_hp=normpdf(samples,priorMean,sqrt(width^2+priorSD^2))';

            if want_posteriors

                if any(isnan(locations_hp)) % this hyperparameter is to be marginalised
                    n_locations_samples_hp = repmat(ns_hp,num_locations,1);%matrify(@(x,y) normpdf(x,y,sqrt(width^2+priorSD^2)),priorMean*ones(num_locations,1),samples);
                else
                    n_locations_samples_hp = matrify(@(x,y) normpdf(x,y,width).*normpdf(x,priorMean,priorSD),locations_hp,samples);
                end
                n_locations_samples = n_locations_samples.*n_locations_samples_hp;
            end
        case 'bounded'
            
            Upper = priorMean+priorSD;
            Lower = priorMean-priorSD;
            
            ns_hp=(normcdf(Upper,samples,width)-normcdf(Lower,samples,width))'/(Upper-Lower);
            

            if want_posteriors
                if any(isnan(locations_hp)) % this hyperparameter is to be marginalised
                    n_locations_samples_hp = repmat(ns_hp,num_locations,1);%matrify(@(x,y) normcdf(Upper,y,width)-normcdf(Lower,y,width),ones(num_locations,1),samples);
                else
                    n_locations_samples_hp = matrify(@(x,y) normpdf(x,y,width).*(Lower<x).*(Upper>x)/(Upper-Lower),locations_hp,samples);
                end
                n_locations_samples = n_locations_samples.*n_locations_samples_hp;
            end
    end
    
    
    
    ns = ns.*ns_hp;
    

    germank_hp = repmat(ns_hp,num_hyperparams,1);
    exp_germank_hp = germank_hp;
    switch type
        case 'real'

            germank_hp(hyperparam,:) = germank_hp(hyperparam,:) .* ...
                (width^2*priorMean+priorSD^2*samples')/(priorSD^2+width^2);
            exp_germank_hp(hyperparam,:) = exp_germank_hp(hyperparam,:) .* ...
                exp((width^2*priorMean+priorSD^2*samples'+0.5*width^2*priorSD^2)/(priorSD^2+width^2));
            
            
        case 'bounded'
            
            germank_hp(hyperparam,:) = germank_hp(hyperparam,:) .* samples' - width^2*(normpdf(Upper,samples',width)-normpdf(Lower,samples',width))/(Upper-Lower);
    end
    
    germank = germank.*germank_hp;
    exp_germank = exp_germank.*exp_germank_hp;

    Ks_hp = matrify(@(x,y) normpdf(x,y,width),samples,samples);
    Ks = Ks.*Ks_hp;
    
end

chol_Ks = chol(Ks);
    
mean_numer = solve_chol(chol_Ks,germank')';
exp_mean_numer = solve_chol(chol_Ks,exp_germank')';
denom = solve_chol(chol_Ks,ns')';

[logLcell{1:num_samples}]=covvy.hypersamples(:).logL;
logLvec=cat(1,logLcell{:});

% [glogLcell{1:num_samples}]=covvy.hypersamples(:).glogL; % actually glogl is a cell itself
% glogLmat=cat(1,logLcell{:});
% glogLvec=reshape(glogLmat,[],1);

logLvec=(logLvec-max(logLvec)); 


% datatwothirds=solve_chol(chol_Ks,exp(logLvec));
% denominator=(ns*datatwothirds);

%postmean = (germank1.*repmat(germank2,num_hyperparams,1))*datatwothirds/denominator;
%postmean=(allcombs(B)'.*kron2d(ones(num_hyperparams,1),A))*datahalf/denominator;
            
denominator = denom*exp(logLvec);

postmean = mean_numer*exp(logLvec)/denominator;
exp_postmean = exp_mean_numer*exp(logLvec)/denominator;
        
if want_posteriors
    post_numer = solve_chol(chol_Ks,n_locations_samples')';
    posteriors = post_numer*exp(logLvec)/denominator;
    
    exp_posteriors = posteriors./exp_locations;
end


        
function s = separation(ls) 
if length(ls)<=1
    s=1;
else
    s=(max(ls)-min(ls))/(length(ls)-1);
end