function covvy=bmcparams(covvy)
% Use MAP for the width parameters of our GP fit - requires checking out
% the logL's in covvy. Also means I'm just going to flat out recompute
% everything below everytime this is called - actually not if there's only
% a single sample


if isfield(covvy, 'derivs_cov')
    derivs = covvy.derivs_cov;
else
    try
        [K DK] = covvy.covfn(covvy.hypersamples(1).hyperparameters, ...
                                                        'deriv hyperparams');
        derivs=true;
    catch
        derivs=false;
    end
end

if isfield(covvy, 'store_full_kron_prod')
    store_full_kron_prod = covvy.store_full_kron_prod;
else
    store_full_kron_prod = true;
    covvy.store_full_kron_prod = store_full_kron_prod;
end

lowr.UT=true;
lowr.TRANSA=true;
uppr.UT=true;

%allow for possibility of either integer or real hyperparams

num_hps=numel(covvy.hyperparams);
hps=1:num_hps;

if ~isfield(covvy,'widthfrac')
covvy.widthfrac=0.20;
end
widthfrac=covvy.widthfrac;

if derivs


    for hyperparam=hps
        
        IndepSamples=covvy.hyperparams(hyperparam).samples;
        priorMean=covvy.hyperparams(hyperparam).priorMean;
        priorSD=covvy.hyperparams(hyperparam).priorSD;
        
        if ~all(~isnan([IndepSamples;priorMean;priorSD])) ...
                || strcmp(covvy.hyperparams(hyperparam).type,'inactive');       
            % This hyperparameter is a dummy - ignore it
            hps=setdiff(hps,hyperparam);
        end
    end
    
    KSinv_NS=ones(1,length(hps)+1);
    cholKsL=1;
    NSamplesPrev=1;
    notignored=0;  

    for hyperparam=hps
        
        type=covvy.hyperparams(hyperparam).type;
        

        IndepSamples=covvy.hyperparams(hyperparam).samples;
        NIndepSamples=covvy.hyperparams(hyperparam).NSamples;
        priorMean=covvy.hyperparams(hyperparam).priorMean;
        priorSD=covvy.hyperparams(hyperparam).priorSD;

        NSamples=NSamplesPrev*NIndepSamples;

        notignored=notignored+1;

        width=widthfrac*separation(IndepSamples);
        KsQ=matrify(@(x,y) normpdf(x,y,width),IndepSamples,IndepSamples);
        KsQ = improve_covariance_conditioning(KsQ);
        
        cholKsQ=chol(KsQ);

        DKs=matrify(@(x,y) (x-y)/width^2.*...
            normpdf(x,y,width),IndepSamples,IndepSamples); 
        % the variable you're taking the derivative wrt is negative
        DKsD=matrify(@(x,y) 1/width^2*(1-((x-y)/width).^2).*...
            normpdf(x,y,width),IndepSamples,IndepSamples);

        KsL=[KsQ,DKs;-DKs,DKsD];
        
        
        cholKsL=kron2d(cholKsL,updatechol(KsL,cholKsQ,NIndepSamples+1:length(KsL)));
        remove=4; 
        % This is not a parameter - equal to 2^2 due to structure 
        % of cov matrix
        for ind=1:notignored-1
            cholKsL=downdatechol(cholKsL,(remove-1)*NSamples+1:remove*NSamples);
            remove=remove+1;
        end

        switch lower(type)
            case 'real'
                PrecX=(priorSD^2+width^2-priorSD^4/(priorSD^2+width^2))^(-1);
                PrecY=(priorSD^2-(priorSD^2+width^2)^2/(priorSD^2))^(-1);
            %     NSfn2=@(x,y) mvnpdf([x;y],[SamplesMean(d);SamplesMean(d)],...
            %         [SamplesSD(d)^2+width^2,SamplesSD(d)^2;SamplesSD(d)^2,SamplesSD(d)^2+width^2]);
                const = (4*pi^2*(priorSD^2+width^2)/PrecX)^(-0.5);
                NSfn=@(x,y) const*...
                    exp(-0.5*PrecX*((x-priorMean).^2+(y-priorMean).^2)-...
                    PrecY.*(x-priorMean).*(y-priorMean));
                
                BS=-width^-2*(repmat(IndepSamples',NIndepSamples,1)-priorMean-...
                    matrify(@(x,y) (PrecX+PrecY)*priorSD^2*...
                    ((x-priorMean)+(y-priorMean)),IndepSamples,IndepSamples));
                
            case 'bounded'
                NSfn=@(x,y) normpdf(x,y,sqrt(2)*width).*...
                    (normcdf(priorMean-priorSD,0.5*(x+y),sqrt(0.5)*width)...
                    -normcdf(priorMean+priorSD,0.5*(x+y),sqrt(0.5)*width));
            case 'mixture'
                mixtureWeights = covvy.hyperparams(hyperparam).mixtureWeights;

                PrecX=(priorSD.^2+width^2-priorSD.^4./(priorSD.^2+width^2)).^(-1);
                PrecY=(priorSD.^2-(priorSD.^2+width^2)^2./(priorSD.^2)).^(-1);
                const = (4*pi^2*(priorSD.^2+width^2)./PrecX).^(-0.5);

                NSfn = @(x,y) mixture_NSfn(x,y,mixtureWeights,const,...
                    PrecX,PrecY,priorMean);

        end
        NS=...%diag(normpdf(IndepSamples,SamplesMean(d),SamplesSD(d)));
        matrify(NSfn,IndepSamples,IndepSamples);    


        % NB: (PrecX+PrecY)*priorSD^2 == priorSD^2/(width^2+2*priorSD^2)

        
        
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
    
elseif ~derivs
    
    if store_full_kron_prod
        KSinv_NS_KSinv=1;
        cholKS = 1;
    else
        KSinv_NS_KSinv=cell(num_hps,1);
        cholKS = cell(num_hps,1);
    end

    
    for hyperparam=hps
        
        
        type=covvy.hyperparams(hyperparam).type;
        
        
        if strcmp(type,'discrete')
            
            % For discrete, we assume we have a sample at every position.
            
            priors = covvy.hyperparams(hyperparam).priors;
            
            KSinv_NS_KSinv_hp = diag(priors);
            
            IndepSamples=covvy.hyperparams(hyperparam).samples;
            width=widthfrac*separation(IndepSamples);
            KsQ_hp=matrify(@(x,y) normpdf(x,y,width),IndepSamples,IndepSamples);
            cholKsQ_hp=chol(KsQ_hp);
            
            if store_full_kron_prod
                KSinv_NS_KSinv = kron2d(KSinv_NS_KSinv,KSinv_NS_KSinv_hp);
                cholKS = kron2d(cholKS,cholKsQ_hp);
            else
                KSinv_NS_KSinv{hyperparam}=KSinv_NS_KSinv_hp;
                cholKS{hyperparam} = cholKsQ_hp;
            end
            
        else
        
            IndepSamples=covvy.hyperparams(hyperparam).samples;
            NIndepSamples=covvy.hyperparams(hyperparam).NSamples;
            priorMean=covvy.hyperparams(hyperparam).priorMean;
            priorSD=covvy.hyperparams(hyperparam).priorSD;

            if ~all(~isnan([IndepSamples;priorMean;priorSD])) ...
                || strcmp(type,'inactive');     
                % This hyperparameter is a dummy - ignore it
                continue
            end

            width=widthfrac*separation(IndepSamples);
        %     fmincon(@(activeTheta) negLogLikelihood(exp(logL),IndepSamples,activeTheta,active,Priorwidth),...
        %         MLwidth(active),[],[],[],[],zeros(length(active),1),largeLimit,[],optimoptions);
            KsQ=matrify(@(x,y) normpdf(x,y,width),IndepSamples,IndepSamples);
            KsQ = improve_covariance_conditioning(KsQ);
            cholKsQ=chol(KsQ);

            cholKsL=cholKsQ;

        %     DKs=matrify(@(x,y) (x-y)/width^2.*normpdf(x,y,width),IndepSamples,IndepSamples); % the variable you're taking the derivative wrt is negative
        %     DKsD=matrify(@(x,y) 1/width^2*(1-((x-y)/width).^2).*normpdf(x,y,width),IndepSamples,IndepSamples);

        %     KsL=[KsQ,DKs;-DKs,DKsD];
        %     cholKsL=updatechol(KsL,KsQ,NIndepSamples+1:length(KsL));

            switch lower(type)
                case 'real'
                    PrecX=(priorSD^2+width^2-priorSD^4/(priorSD^2+width^2))^(-1);
                    PrecY=(priorSD^2-(priorSD^2+width^2)^2/(priorSD^2))^(-1);
                %     NSfn2=@(x,y) mvnpdf([x;y],[SamplesMean(d);SamplesMean(d)],...
                %         [SamplesSD(d)^2+width^2,SamplesSD(d)^2;SamplesSD(d)^2,SamplesSD(d)^2+width^2]);
                    const = (4*pi^2*(priorSD^2+width^2)/PrecX)^(-0.5);
                    NSfn=@(x,y) const*...
                        exp(-0.5*PrecX*((x-priorMean).^2+(y-priorMean).^2)-...
                        PrecY.*(x-priorMean).*(y-priorMean));
                case 'bounded'
                    NSfn=@(x,y) normpdf(x,y,sqrt(2)*width).*...
                        (normcdf(priorMean-priorSD,0.5*(x+y),sqrt(0.5)*width)-normcdf(priorMean+priorSD,0.5*(x+y),sqrt(0.5)*width));
                case 'mixture'
                    mixtureWeights = covvy.hyperparams(hyperparam).mixtureWeights;
                    
                    PrecX=(priorSD.^2+width^2-priorSD.^4./(priorSD.^2+width^2)).^(-1);
                    PrecY=(priorSD.^2-(priorSD.^2+width^2)^2./(priorSD.^2)).^(-1);
                    const = (4*pi^2*(priorSD.^2+width^2)./PrecX).^(-0.5);
                    
                    NSfn = @(x,y) mixture_NSfn(x,y,mixtureWeights,const,PrecX,PrecY,priorMean);
                    
            end
            
            
            NS=...%diag(normpdf(IndepSamples,SamplesMean(d),SamplesSD(d)));
                    matrify(NSfn,IndepSamples,IndepSamples);

                
            KSinv_NS_KSinv_hp=cholKsQ\(cholKsQ'\((NS/cholKsL)/cholKsL'));
                
            if store_full_kron_prod
                KSinv_NS_KSinv = kron2d(KSinv_NS_KSinv,KSinv_NS_KSinv_hp);
                cholKS = kron2d(cholKS,cholKsQ);
            else
                KSinv_NS_KSinv{hyperparam}=KSinv_NS_KSinv_hp;
                cholKS{hyperparam} = cholKsQ;
            end
        end


    end

    covvy.KSinv_NS_KSinv=KSinv_NS_KSinv;
    covvy.cholKS=cholKS;
end




function s = separation(ls) 
if length(ls)<=1
    s=1;
else
    s=(max(ls)-min(ls))/(length(ls)-1);
end

function NS = mixture_NSfn(x,y,weights,const,PrecX,PrecY,priorMean)

NS = 0;
for i = 1:length(weights)
    NS = NS + const(i)*...
        exp(-0.5*PrecX(i)*((x-priorMean(i)).^2+(y-priorMean(i)).^2)-...
        PrecY(i).*(x-priorMean(i)).*(y-priorMean(i)));
end
