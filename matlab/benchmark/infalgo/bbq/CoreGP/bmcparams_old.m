function covvyout=bmcparams(covvy,flag)
% Use MAP for the width parameters of our GP fit - requires checking out
% the logL's in covvy. Also means I'm just going to flat out recompute
% everything below everytime this is called - actually not if there's only
% a single sample

%'flag'=do you want MS term#

if nargin<2
    flag='MeanOnly';
end

if ((isfield(covvy, 'use_derivatives') && covvy.use_derivatives == ...
						 true) || (isfield(covvy, 'covfn') && nargin(covvy.covfn)~=1))
    derivs=true;
else
    % we can determine the gradient of the covariance wrt hyperparams
    derivs=false;
end

covvyout=covvy;

lowr.UT=true;
lowr.TRANSA=true;
uppr.UT=true;

%allow for possibility of either integer or real hyperparams

TSamples=numel(covvy.hyperparams);
hps=1:TSamples;

widthfrac=0.20;

if derivs


    for hyperparam=hps
        
        IndepSamples=covvy.hyperparams(hyperparam).samples;
        priorMean=covvy.hyperparams(hyperparam).priorMean;
        priorSD=covvy.hyperparams(hyperparam).priorSD;
        
        if ~all(~isnan([IndepSamples;priorMean;priorSD]));       
            % This hyperparameter is a dummy - ignore it
            hps=setdiff(hps,hyperparam);
        end
    end
    
    KSinv_NS=ones(1,length(hps)+1);
    cholKsL=1;
    MsTerm=1;
    NsTerm=1;
    NSamplesPrev=1;
    notignored=0;  

    for hyperparam=hps

        IndepSamples=covvy.hyperparams(hyperparam).samples;
        NIndepSamples=covvy.hyperparams(hyperparam).NSamples;
        priorMean=covvy.hyperparams(hyperparam).priorMean;
        priorSD=covvy.hyperparams(hyperparam).priorSD;

        NSamples=NSamplesPrev*NIndepSamples;

        notignored=notignored+1;

        width=widthfrac*separation(IndepSamples);
    %     fmincon(@(activeTheta) negLogLikelihood(exp(logL),IndepSamples,activeTheta,active,Priorwidth),...
    %         MLwidth(active),[],[],[],[],zeros(length(active),1),largeLimit,[],optimoptions);
        KsQ=matrify(@(x,y) normpdf(x,y,width),IndepSamples,IndepSamples);
        cholKsQ=chol(KsQ);

        DKs=matrify(@(x,y) (x-y)/width^2.*normpdf(x,y,width),IndepSamples,IndepSamples); % the variable you're taking the derivative wrt is negative
        DKsD=matrify(@(x,y) 1/width^2*(1-((x-y)/width).^2).*normpdf(x,y,width),IndepSamples,IndepSamples);

        KsL=[KsQ,DKs;-DKs,DKsD];
        
        
        cholKsL=kron2d(cholKsL,updatechol(KsL,cholKsQ,NIndepSamples+1:length(KsL)));
        remove=4; % This is not a parameter - equal to 2^2 due to structure of cov matrix
        for ind=1:notignored-1
            cholKsL=downdatechol(cholKsL,(remove-1)*NSamples+1:remove*NSamples);
            remove=remove+1;
        end

        PrecX=(priorSD^2+width^2-priorSD^4/(priorSD^2+width^2))^(-1);
        PrecY=(priorSD^2-(priorSD^2+width^2)^2/(priorSD^2))^(-1);
    %     NSfn2=@(x,y) mvnpdf([x;y],[SamplesMean(d);SamplesMean(d)],...
    %         [SamplesSD(d)^2+width^2,SamplesSD(d)^2;SamplesSD(d)^2,SamplesSD(d)^2+width^2]);
        NSfn=@(x,y) (4*pi^2*(priorSD^2+width^2)/PrecX)^(-0.5)*...
            exp(-0.5*PrecX*((x-priorMean).^2+(y-priorMean).^2)-...
            PrecY.*(x-priorMean).*(y-priorMean));
        NS=...%diag(normpdf(IndepSamples,SamplesMean(d),SamplesSD(d)));
        matrify(NSfn,IndepSamples,IndepSamples);    

        BS=-width^-2*(repmat(IndepSamples',NIndepSamples,1)-priorMean-...
            matrify(@(x,y) (PrecX+PrecY)*priorSD^2*((x-priorMean)+(y-priorMean)),IndepSamples,IndepSamples));
        % NB: (PrecX+PrecY)*priorSD^2 == priorSD^2/(width^2+2*priorSD^2)

        
        
        left=(length(hps)+1-notignored)*NSamplesPrev; % derivs are listed in reverse order
        right=(length(hps)+1-notignored+1)*NSamplesPrev; % NB: first block of  KSinv_NS has nothing to do with derivs
        
        KSinv_NS_new=cholKsQ\(cholKsQ'\NS);
        
        KSinv_NS=...    
            [kron2d(KSinv_NS(:,1:left),KSinv_NS_new),...
            kron2d(KSinv_NS(:,left+1:right),cholKsQ\(cholKsQ'\(BS.*NS))),...
            kron2d(KSinv_NS(:,right+1:end),KSinv_NS_new)];


        if strcmp(flag,'VarianceTerms')

             ns=normpdf(IndepSamples,priorMean,sqrt(width^2+priorSD^2));
        %     MsSD(d)=sqrt(SamplesSD(d)^(-2)*(SamplesSD(d)^2+width^2)...
        %         *(width^2+2*(SamplesSD(d)^2-SamplesSD(d)^2*(SamplesSD(d)^2+width^2)^(-1)*SamplesSD(d)^2))...
        %         *(SamplesSD(d)^2+width^2)*SamplesSD(d)^(-2));
        %     %tends to zero as width tends to zero
             MS=(ns*ns').*normpdf((1-width^2*(priorSD^2+width^2)^(-1))*matrify(@minus,IndepSamples,IndepSamples),...
                    zeros(NIndepSamples),...
                    sqrt(3*width^2-2*width^2*(priorSD^2+width^2)^(-1)*width^2)*ones(NIndepSamples));

    %          MsTerm=kron2d(MsTerm,(Ks\MS)/Ks);
    %         NsTerm=kron2d(NsTerm,inv(Ks)*NS*inv(Ks)*NS*inv(Ks)); 

            MsTerm=kron2d(MsTerm,...
                linsolve(cholKsL,...
                linsolve(cholKsL,...
                linsolve(cholKsL,...
                linsolve(cholKsL,MS,...
                lowr),uppr)',lowr),uppr)');

                HalfNsTerm=linsolve(cholKsQ,...
                    linsolve(cholKsL,...
                    linsolve(cholKsL,NS,...
                    lowr),uppr)',lowr);

               NsTerm=kron2d(NsTerm,HalfNsTerm'*HalfNsTerm);

        end

        %    MS=diag(normpdf(IndepSamples,SamplesMean(d),priorSD).^2);

        %     for i=1:length(IndepSamples)
        %         for j=1:length(IndepSamples)
        %             MSfn(i,j)=normpdf(IndepSamples(i),SamplesMean(d),sqrt(width^2+priorSD^2))*...
        %                 mvnpdf([0;IndepSamples(j)],[priorSD^2*inv(priorSD^2+width^2)*(IndepSamples(i)-SamplesMean(d));SamplesMean(d)],...
        %                 [width^2+2*priorSD^2-priorSD^2*inv(priorSD^2+width^2)*priorSD^2,priorSD^2;...
        %                 priorSD^2,priorSD^2+width^2]);
        %         end
        %     end

        NSamplesPrev=NSamples;
    end
   

    covvyout.KSinv_NS_KSinv=(KSinv_NS/cholKsL)/cholKsL';
    
else
    
    KSinv_NS_KSinv=1;
    MsTerm=1;
    NsTerm=1;
    for hyperparam=hps
        
        
        type=covvy.hyperparams(hyperparam).type;
        
        
        if strcmp(type,'discrete')
            
            % For discrete, we assume we have a sample at every position.
            
            priors = covvy.hyperparams(hyperparam).priors;
            
            KSinv_NS_KSinv=...%kron2d(KSinv_NS_KSinv,(Ks\NS)/Ks);
            kron2d(KSinv_NS_KSinv,diag(priors));
            
        else
        
            IndepSamples=covvy.hyperparams(hyperparam).samples;
            NIndepSamples=covvy.hyperparams(hyperparam).NSamples;
            priorMean=covvy.hyperparams(hyperparam).priorMean;
            priorSD=covvy.hyperparams(hyperparam).priorSD;

            if ~all(~isnan([IndepSamples;priorMean;priorSD]));
                % This hyperparameter is a dummy - ignore it
                continue
            end

            width=widthfrac*separation(IndepSamples);
        %     fmincon(@(activeTheta) negLogLikelihood(exp(logL),IndepSamples,activeTheta,active,Priorwidth),...
        %         MLwidth(active),[],[],[],[],zeros(length(active),1),largeLimit,[],optimoptions);
            KsQ=matrify(@(x,y) normpdf(x,y,width),IndepSamples,IndepSamples);
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
                    weights = covvy.hyperparams(hyperparam).mixtureWeights;
                    
                    PrecX=(priorSD.^2+width^2-priorSD.^4./(priorSD.^2+width^2)).^(-1);
                    PrecY=(priorSD.^2-(priorSD.^2+width^2)^2./(priorSD.^2)).^(-1);
                    const = (4*pi^2*(priorSD.^2+width^2)./PrecX).^(-0.5);
                    
                    NSfn = @(x,y) mixture_NSfn(x,y,weights,const,PrecX,PrecY,priorMean);
                    
            end
            NS=...%diag(normpdf(IndepSamples,SamplesMean(d),SamplesSD(d)));
                    matrify(NSfn,IndepSamples,IndepSamples);



            KSinv_NS_KSinv=...%kron2d(KSinv_NS_KSinv,(Ks\NS)/Ks);
            kron2d(KSinv_NS_KSinv,cholKsQ\(cholKsQ'\((NS/cholKsL)/cholKsL')));
        end

        if strcmp(flag,'VarianceTerms')
        switch lower(type)
            case 'real'

             ns=normpdf(IndepSamples,priorMean,sqrt(width^2+priorSD^2));
        %     MsSD(d)=sqrt(SamplesSD(d)^(-2)*(SamplesSD(d)^2+width^2)...
        %         *(width^2+2*(SamplesSD(d)^2-SamplesSD(d)^2*(SamplesSD(d)^2+width^2)^(-1)*SamplesSD(d)^2))...
        %         *(SamplesSD(d)^2+width^2)*SamplesSD(d)^(-2));
        %     %tends to zero as width tends to zero
             MS=(ns*ns').*normpdf((1-width^2*(priorSD^2+width^2)^(-1))*matrify(@minus,IndepSamples,IndepSamples),...
                    zeros(NIndepSamples),...
                    sqrt(3*width^2-2*width^2*(priorSD^2+width^2)^(-1)*width^2)*ones(NIndepSamples));

    %          MsTerm=kron2d(MsTerm,(Ks\MS)/Ks);
    %         NsTerm=kron2d(NsTerm,inv(Ks)*NS*inv(Ks)*NS*inv(Ks)); 

            MsTerm=kron2d(MsTerm,...
                linsolve(cholKsL,...
                linsolve(cholKsL,...
                linsolve(cholKsL,...
                linsolve(cholKsL,MS,...
                lowr),uppr)',lowr),uppr)');

            HalfNsTerm=linsolve(cholKsQ,...
                linsolve(cholKsL,...
                linsolve(cholKsL,NS,...
                lowr),uppr)',lowr);

           NsTerm=kron2d(NsTerm,HalfNsTerm'*HalfNsTerm);
        end

        end

    %    MS=diag(normpdf(IndepSamples,SamplesMean(d),priorSD).^2);

    %     for i=1:length(IndepSamples)
    %         for j=1:length(IndepSamples)
    %             MSfn(i,j)=normpdf(IndepSamples(i),SamplesMean(d),sqrt(width^2+priorSD^2))*...
    %                 mvnpdf([0;IndepSamples(j)],[priorSD^2*inv(priorSD^2+width^2)*(IndepSamples(i)-SamplesMean(d));SamplesMean(d)],...
    %                 [width^2+2*priorSD^2-priorSD^2*inv(priorSD^2+width^2)*priorSD^2,priorSD^2;...
    %                 priorSD^2,priorSD^2+width^2]);
    %         end
    %     end


    end

    covvyout.KSinv_NS_KSinv=KSinv_NS_KSinv;
end

if strcmp(flag,'VarianceTerms')    
    covvyout.MsTerm=MsTerm;
    covvyout.NsTerm=NsTerm;
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
