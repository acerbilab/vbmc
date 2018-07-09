function covvy=posterior_hp_params(covvy,locations)
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

if isfield(covvy, 'store_full_kron_prod')
    store_full_kron_prod = covvy.store_full_kron_prod;
else
    store_full_kron_prod = true;
    covvy.store_full_kron_prod = store_full_kron_prod;
end

num_hyperparams=numel(covvy.hyperparams);
num_samples=numel(covvy.hypersamples);

want_posteriors=nargin>1 && nargout>1;
want_exp = nargout>2;

chol_Ks = cell(num_hyperparams,1);
ns = 1;
germank = ones(num_hyperparams,1);
if want_exp
    exp_germank = ones(num_hyperparams,1);
end

if want_posteriors
    num_locations = size(locations,1);
    n_locations_samples = ones(num_locations,1);
    if want_exp
        exp_locations = ones(num_locations,1);
    end
end

if isfield(covvy,'hyper2samples') && isfield(covvy,'ML_hyper2sample_ind')
    widths = exp(covvy.hyper2samples(covvy.ML_hyper2sample_ind).hyper2parameters);
else
    widths = nan(1,num_hyperparams);
    if ~isfield(covvy,'widthfrac')
    covvy.widthfrac=0.20;
    end
    widthfrac=covvy.widthfrac;
    
    for hyperparam=1:num_hyperparams
        samples = covvy.hyperparams(hyperparam).samples;
        widths(hyperparam)=widthfrac*separation(samples);
    end
end



for hyperparam=1:num_hyperparams
    
    samples = covvy.hyperparams(hyperparam).samples;
    NIndepSamples=covvy.hyperparams(hyperparam).NSamples;
    priorMean=covvy.hyperparams(hyperparam).priorMean;
    priorSD=covvy.hyperparams(hyperparam).priorSD;
    if isfield(covvy.hyperparams,'type')
        type  = covvy.hyperparams(hyperparam).type;
    else
        type = 'real';
    end
    if ~all(~isnan([samples;priorMean;priorSD])) %|| NIndepSamples==1
        type = 'inactive';
    end
    
    switch type
        case 'inactive'
            % This hyperparameter is a dummy - ignore it
            continue
    end
    
    
    width=widths(hyperparam);
    
    if want_posteriors
        locations_hp = locations(:,hyperparam);
        if want_exp && strcmp(type,'real') && ~any(isnan(locations_hp))
            exp_locations = exp_locations.*exp(locations_hp);
        end
    end
    
    Ks_base_hp = matrify(@(x,y) normpdf(x,y,width),samples,samples);
    
    %     Ks=Ks_base;
    %     jitterstep=0.05*Ks_base(1);
    %     jitter=jitterstep;
    %     while cond(Ks)>100
    %         Ks=Ks_base+jitter.^2;
    %         jitter=jitter+jitterstep;
    %     end
    %     jitter=jitter-jitterstep;
    %     Ks=Ks_base+jitter.^2;
    %     cholKs=chol(Ks);
    
    chol_Ks_hp = chol(Ks_base_hp);
    chol_Ks{hyperparam} = chol_Ks_hp;
    
    switch type
        case 'real'
            ns_hp=normpdf(samples,priorMean,sqrt(width^2+priorSD^2))';

            if want_posteriors
                if any(isnan(locations_hp)) % this hyperparameter is to be marginalised
                    n_locations_samples_hp = repmat(ns_hp,num_locations,1);%matrify(@(x,y) normpdf(x,y,sqrt(width^2+priorSD^2)),priorMean*ones(num_locations,1),samples);
                else
                    n_locations_samples_hp = matrify(@(x,y) normpdf(x,y,width).*normpdf(x,priorMean,priorSD),locations_hp,samples);
                end
            end
        case 'bounded'

            Upper = priorMean+priorSD;
            Lower = priorMean-priorSD;

            ns_hp=(normcdf(Upper,samples,width)-normcdf(Lower,samples,width))'/(Upper-Lower);


            if want_posteriors
                if any(isnan(locations_hp)) % this hyperparameter is to be marginalised
                    n_locations_samples_hp = repmat(ns_hp,num_locations,1);
                else
                    n_locations_samples_hp = matrify(@(x,y) normpdf(x,y,width).*...
                        (Lower<x).*(Upper>x)/(Upper-Lower),locations_hp,samples);
                end

            end

        case 'mixture'
            weights = covvy.hyperparams(hyperparam).mixtureWeights;

            ns_hp = 0;
            for i=1:length(weights)
                ns_hp = ns_hp + weights(i) * normpdf(samples,priorMean(i),sqrt(width^2+priorSD(i)^2))';
            end

            if want_posteriors
                if any(isnan(locations_hp)) % this hyperparameter is to be marginalised
                    n_locations_samples_hp = repmat(ns_hp,num_locations,1);
                else
                    n_locations_samples_hp = 0;
                    for i = 1:length(weights)
                        n_locations_samples_hp = n_locations_samples_hp + ...
                            weights(i) * matrify(@(x,y) normpdf(x,y,width).*...
                            normpdf(x,priorMean(i),priorSD(i)),locations_hp,samples);
                    end
                end
            end
        case 'discrete'
            priors = covvy.hyperparams(hyperparam).priors;
            if size(priors,2) == 1
                priors = priors';
            end
            mat = matrify(@(x,y) normpdf(x,y,width),samples,samples);

            ns_hp = priors*mat;

            if want_posteriors
                if any(isnan(locations_hp)) % this hyperparameter is to be marginalised
                    n_locations_samples_hp = repmat(ns_hp,num_locations,1);
                else
                    priors = priors';
                    loc_vec = nan(length(locations_hp),1);
                    for i=1:length(locations_hp)
                        loc_vec(i) = find(locations_hp(i)==samples);
                    end
                    n_locations_samples_hp = matrify(@(x,y) normpdf(x,y,width),locations_hp,samples)...
                        .*repmat(priors(loc_vec),1,length(samples));
                    priors = priors';
                end
            end

    end

    % germank is the term used to calculate the posterior mean
    germank_hp = repmat(ns_hp,num_hyperparams,1);
    if want_exp
        exp_germank_hp = germank_hp;
    end

    switch type
        case 'real'

            germank_hp(hyperparam,:) = germank_hp(hyperparam,:)...
                .* (width^2*priorMean+priorSD^2*samples')/(priorSD^2+width^2);
            if want_exp
                exp_germank_hp(hyperparam,:) = exp_germank_hp(hyperparam,:) .* ...
                    exp((width^2*priorMean+priorSD^2*samples'+0.5*width^2*priorSD^2)/(priorSD^2+width^2));
            end


        case 'bounded'

            germank_hp(hyperparam,:) = germank_hp(hyperparam,:) .* samples'...
                - width^2*(normpdf(Upper,samples',width)...
                -normpdf(Lower,samples',width))/(Upper-Lower);
            if want_exp
                exp_germank_hp(hyperparam,:) = germank_hp(hyperparam,:);
            end
            
        case 'mixture'

            germank_hp(hyperparam,:) = 0;
            for i=1:length(weights)
                germank_hp(hyperparam,:) = germank_hp(hyperparam,:)...
                    + weights(i)*normpdf(samples,priorMean(i),sqrt(width^2+priorSD(i)^2))'...
                    *(width^2*priorMean(i)+priorSD(i)^2*samples')/(priorSD(i)^2+width(i)^2);
            end

            if want_exp
                exp_germank_hp(hyperparam,:) = 0;
                for i=1:length(weights)
                    exp_germank_hp(hyperparam,:) = exp_germank_hp(hyperparam,:)...
                        + weights(i)*normpdf(samples,priorMean(i),sqrt(width^2+priorSD(i)^2))'...
                        *exp((width^2*priorMean+priorSD^2*samples'+0.5*width^2*priorSD^2)/(priorSD^2+width^2));
                end
            end
            
        case 'discrete'
            
            germank_hp(hyperparam,:) = (priors.*samples')*mat;
            if want_exp
                exp_germank_hp(hyperparam,:) = germank_hp(hyperparam,:);
            end
            
    end

    ns = kron1d(ns,ns_hp);
    germank = kron1d(germank,germank_hp);
    if want_exp
        exp_germank = kron1d(exp_germank,exp_germank_hp);
    end
    if want_posteriors
        n_locations_samples = kron1d(n_locations_samples,n_locations_samples_hp);
    end
end



[logLcell{1:num_samples}]=covvy.hypersamples(:).logL;
logLvec=cat(1,logLcell{:});

% [glogLcell{1:num_samples}]=covvy.hypersamples(:).glogL; % actually glogl is a cell itself
% glogLmat=cat(1,logLcell{:});
% glogLvec=reshape(glogLmat,[],1);

logLvec=(logLvec-max(logLvec)); 
exp_logLvec = exp(logLvec);
            
invK_L = kron_solve_chol(chol_Ks,exp_logLvec);
denominator = ns*invK_L;

invK_L_invdenom = invK_L/denominator;


postmean = germank*invK_L_invdenom;
if want_exp
    exp_postmean = exp_germank*invK_L_invdenom;
end
        
if want_posteriors
    posteriors = n_locations_samples*invK_L_invdenom;
    
    if want_exp
        exp_posteriors = posteriors./exp_locations;
    end
end


        
function s = separation(ls) 
if length(ls)<=1
    s=1;
else
    s=(max(ls)-min(ls))/(length(ls)-1);
end