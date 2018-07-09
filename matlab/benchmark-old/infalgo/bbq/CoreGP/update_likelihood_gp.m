function [likelihood_gp] = update_likelihood_gp(gp,likelihood_gp)


active_hp_inds = gp.active_hp_inds;

hypersamples = [gp.hypersamples.hyperparameters];
hypersamples = hypersamples(:,active_hp_inds);

num_hps = size(hypersamples,2);
num_data = size(hypersamples,1);

prior_means = [gp.hyperparams.priorMean];
prior_means = prior_means(:,active_hp_inds);

prior_SDs = [gp.hyperparams.priorSD];
prior_SDs = prior_SDs(:,active_hp_inds);

if nargin<2
    % we 
    if size(hypersamples,1) == 1
        hp_scales = ones(1,num_hps);
    else
        hp_scales = std(hypersamples);
    end
else
    best_quad_hypersample_ind = max([likelihood_gp.hypersamples(:).logL]);
    best_quad_hypersample = ...
        likelihood_gp.hypersamples(best_quad_hypersample_ind).hyperparameters;
    
    hps_struct = set_hps_struct(likelihood_gp);
    hp_scales = exp(best_quad_hypersample(hps_struct.logInputScales));
end

K = ones(num_data);
N = ones(num_data);
for hp = 1:num_hps
    
    width = hp_scales(hp);
    prior_SD_hp = prior_SDs(hp);
    prior_mean_hp = prior_means(hp);
    hypersamples_hp = hypersamples(:,hp);
    
    K_hp = matrify(@(x,y) normpdf(x,y,width), hypersamples_hp, hypersamples_hp);

    PrecX=(prior_SD_hp^2+width^2-prior_SD_hp^4/(prior_SD_hp^2+width^2))^(-1);
    PrecY=(prior_SD_hp^2-(prior_SD_hp^2+width^2)^2/(prior_SD_hp^2))^(-1);
    const = (4*pi^2*(prior_SD_hp^2+width^2)/PrecX)^(-0.5);
    
    N_fn=@(x,y) const*...
        exp(-0.5*PrecX*((x-prior_mean_hp).^2+(y-prior_mean_hp).^2)-...
                PrecY.*(x-prior_mean_hp).*(y-prior_mean_hp));   
    
    N = N.*matrify(@(x,y) N_fn(x,y), hypers, hypers);
    
    
        DKs=matrify(@(x,y) (x-y)/width^2.*...
            normpdf(x,y,width),IndepSamples,IndepSamples); 
        % the variable you're taking the derivative wrt is negative
        DKsD=matrify(@(x,y) 1/width^2*(1-((x-y)/width).^2).*...
            normpdf(x,y,width),IndepSamples,IndepSamples);

        KsL=[KsQ,DKs;-DKs,DKsD];
        
        
        switch lower(type)
            case 'real'
                PrecX=(prior_SD_hp^2+width^2-prior_SD_hp^4/(prior_SD_hp^2+width^2))^(-1);
                PrecY=(prior_SD_hp^2-(prior_SD_hp^2+width^2)^2/(prior_SD_hp^2))^(-1);
            %     NSfn2=@(x,y) mvnpdf([x;y],[SamplesMean(d);SamplesMean(d)],...
            %         [SamplesSD(d)^2+width^2,SamplesSD(d)^2;SamplesSD(d)^2,SamplesSD(d)^2+width^2]);
                const = (4*pi^2*(prior_SD_hp^2+width^2)/PrecX)^(-0.5);
                NSfn=@(x,y) const*...
                    exp(-0.5*PrecX*((x-prior_mean_hp).^2+(y-prior_mean_hp).^2)-...
                    PrecY.*(x-prior_mean_hp).*(y-prior_mean_hp));
                
                BS=-width^-2*(repmat(IndepSamples',NIndepSamples,1)-prior_mean_hp-...
                    matrify(@(x,y) (PrecX+PrecY)*prior_SD_hp^2*...
                    ((x-prior_mean_hp)+(y-prior_mean_hp)),IndepSamples,IndepSamples));
                
            case 'bounded'
                NSfn=@(x,y) normpdf(x,y,sqrt(2)*width).*...
                    (normcdf(prior_mean_hp-prior_SD_hp,0.5*(x+y),sqrt(0.5)*width)...
                    -normcdf(prior_mean_hp+prior_SD_hp,0.5*(x+y),sqrt(0.5)*width));
            case 'mixture'
                mixtureWeights = covvy.hyperparams(hyperparam).mixtureWeights;

                PrecX=(prior_SD_hp.^2+width^2-prior_SD_hp.^4./(prior_SD_hp.^2+width^2)).^(-1);
                PrecY=(prior_SD_hp.^2-(prior_SD_hp.^2+width^2)^2./(prior_SD_hp.^2)).^(-1);
                const = (4*pi^2*(prior_SD_hp.^2+width^2)./PrecX).^(-0.5);

                NSfn = @(x,y) mixture_NSfn(x,y,mixtureWeights,const,...
                    PrecX,PrecY,prior_mean_hp);

        end
        NS=...%diag(normpdf(IndepSamples,SamplesMean(d),SamplesSD(d)));
        matrify(NSfn,IndepSamples,IndepSamples);    


        % NB: (PrecX+PrecY)*prior_SD_hp^2 == prior_SD_hp^2/(width^2+2*prior_SD_hp^2)

        
        
        left=(length(hps)+1-notignored)*NSamplesPrev; 
        % derivs are listed in reverse order
        right=(length(hps)+1-notignored+1)*NSamplesPrev; 
        % NB: first block of  KSinv_NS has nothing to do with derivs
        
        KSinv_NS_new=cholKsQ\(cholKsQ'\NS);
        
        KSinv_NS=...    
            [kron2d(KSinv_NS(:,1:left),KSinv_NS_new),...
            kron2d(KSinv_NS(:,left+1:right),cholKsQ\(cholKsQ'\(BS.*NS))),...
            kron2d(KSinv_NS(:,right+1:end),KSinv_NS_new)];



        NSamplesPrev=NSamples;
    end
   

    covvy.KSinv_NS_KSinv=(KSinv_NS/cholKsL)/cholKsL';
end

K = improve_covariance_conditioning(K);
R = chol(K);
KSinv_NS_KSinv = solve_chol(R,N);
KSinv_NS_KSinv = solve_chol(R,KSinv_NS_KSinv');

gp.KSinv_NS_KSinv = KSinv_NS_KSinv;