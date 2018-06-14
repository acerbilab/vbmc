function [m,C,gm,gC,Hm,HC] = gpmeancov(XStar,XData,covvy,sample,flag)

if nargin<4
    sample=1;
end
% If nocov, we do not compute the covariance
if nargin<5 && nargout>1
    nocov=0;
    nomean=0;
    varnotcov=0;
    condition_corrected_variance=0;
elseif nargout~=1 && strcmp(flag,'condition_corrected_variance') 
    % include extra noise in YStar to reflect what would
    % need to be added to counter conditioning errors should YStar
    % actually be added to YData.
    nocov=0;
    nomean=0;
    varnotcov=0;
    condition_corrected_variance=1;
elseif nargout==1 || strcmp(flag,'no_cov') 
    nomean=0;
    nocov=1;
    varnotcov=0;
    condition_corrected_variance=0;
elseif strcmp(flag,'no_mean')
    nomean=1;
    nocov=0;
    varnotcov=0;
    condition_corrected_variance=0;
elseif strcmp(flag,'var_not_cov')
    % actually return a vector of variances in the place of a covariance matrix
    nomean=0;
    nocov=0;
    varnotcov=1;
    condition_corrected_variance=0;
end

if nargout<3
    [K]=covvy.covfn(covvy.hypersamples(sample).hyperparameters);
elseif nargout<5
    [K,DK]=covvy.covfn(covvy.hypersamples(sample).hyperparameters,'deriv hyperparams');
else
    [K,DK,DDK]=covvy.covfn(covvy.hypersamples(sample).hyperparameters,'deriv hyperparams');       
end
%DK will always assume derivative is taken wrt first argument
if varnotcov
    [vecK]=covvy.covfn(covvy.hypersamples(sample).hyperparameters,'vector');
end

NData=size(XData,1);
NStar=size(XStar,1);

%covvy = gpparams(XData,YData,covvy,sample,'refresh');
cholK=covvy.hypersamples(sample).cholK;

if isfield(covvy,'meanfn')
    % a non-constant prior mean is to be specified
    MuFn=covvy.meanfn(covvy.hypersamples(1).hyperparameters);
    Mu=MuFn(XStar);
    
    % If Mu really does have non-zero derivs, then for gm etc. we're going
    % to have to account for the effects on datatwothirds. Too hard right
    % now!
%     try
%         [Mu,DMu]=covvy.meanfn(covvy.hypersamples(1).hyperparameters,'deriv hyperparams');
%         derivs_mean=true;
%     catch
%         derivs_mean=false;
%     end
else
    if ~isfield(covvy,'meanPos')
        names = {covvy.hyperparams(:).name};
        meanPos=cellfun(@(x) strcmp(x,'mean'),names);
    else
        meanPos=covvy.meanPos;
    end
    Mu=covvy.hypersamples(sample).hyperparameters(meanPos);
    if isempty(Mu)
        Mu=0;
    end
end

datatwothirds=covvy.hypersamples(sample).datatwothirds;

lowr.UT=true;
lowr.TRANSA=true;
uppr.UT=true;

% DK(XStar,XData) & DDK(XStar,XData) are going to be cells, whose indices
% are over the different possible derivatives (there are NVariables in
% total). Each cell content is a matrix DK(XStar,XData) for a different
% possible derivative. This could be done using arrays but I find cells
% more intuitive.

KStarData=K(XStar,XData);

if ~nomean
    m = Mu + KStarData*datatwothirds;     % Compute the objective function value at XStar
end

if nargout>1

    if varnotcov
        F = linsolve(cholK,KStarData',lowr);
        C = vecK(XStar,XStar)-sum(F.^2)'; 
    elseif ~nocov
        Kterm=linsolve(cholK,linsolve(cholK,KStarData',lowr),uppr);

        Kstst = K(XStar,XStar);
        C = Kstst-KStarData*Kterm;
        if condition_corrected_variance
            % I should really find the derivative of the noise added wrt
            % position, but I haven't yet
            
            VStarData = nan(NStar+NData);
            VStarData(1:NStar,(NStar+1):end) = KStarData;
            VStarData((NStar+1):end,1:NStar) = KStarData';
            VStarData(1:NStar,1:NStar) = Kstst;
            VStarData = improve_covariance_conditioning(VStarData);
            
            C = C+VStarData(1:NStar,1:NStar)-Kstst;
        end
        
    end
    
    if nargout>2
        DKStarData=DK(XStar,XData);

        % Gradient of the function evaluated at XStar
        %gm = DK(XStar,XData)*datatwothirds;  
        if ~nomean
            gm=cellfun(@(DKmat) DKmat*datatwothirds,DKStarData,...
                'UniformOutput',false);
        end

        if nargout>3
            %gC = -2*DK(XStar,XData)*Kterm;
            if varnotcov
                
                DF = cellfun(@(DKmat) linsolve(cholK,DKmat',lowr),...
                    DKStarData,...
                    'UniformOutput',false);
                gC = cellfun(@(DFi) -2*sum(DFi.*F)', ...
                    DF,...
                    'UniformOutput',false);
            elseif ~nocov              
                gC=cellfun(@(DKmat) -2*DKmat*Kterm, DKStarData,...
                    'UniformOutput',false);
            end

            if nargout>4

                DDKStarData=DDK(XStar,XData);
                
                %Hm = DDK(XStar,XData)*datatwothirds;  % Hessian evaluated at XStar
                if ~nomean                   
                    Hm=cellfun(@(DDKmat) DDKmat*datatwothirds, DDKStarData,'UniformOutput',false);
                end

                if nargout>5
                    if varnotcov
                        
                        otherterm=linsolve(cholK,cat(1,DKStarData{:})',lowr);
                        
                        % otherterm'*otherterm==
                        % DK(XStar,XData)*inv(K(XData,XData))*DK(XStar,XData)';

                        %HC = -2*DDK(XStar,XData)*Kterm-2*otherterm'*otherterm;

                        HC=-2*reshape(...
                            cat(1,DDKStarData{:})*Kterm,...
                            NVariables*NData,NVariables*NData)-2*otherterm'*otherterm;

                        HC=mat2cell2d(HC,NData*ones(1,NVariables),NData*ones(1,NVariables));
                    elseif ~nocov
                        otherterm=linsolve(cholK,cat(1,DKStarData{:})',lowr);
                        
                        % otherterm'*otherterm==
                        % DK(XStar,XData)*inv(K(XData,XData))*DK(XStar,XData)';

                        %HC = -2*DDK(XStar,XData)*Kterm-2*otherterm'*otherterm;

                        HC=-2*reshape(...
                            cat(1,DDKStarData{:})*Kterm,...
                            NStar*NData,NStar*NData)-2*otherterm'*otherterm;

                        HC=mat2cell2d(HC,NStar*ones(1,NData),NStar*ones(1,NData));
                    end
                end
            end
        end
    end
end

if nocov
    C=[];
    gC=[];
    HC=[];
end
if nomean
    m=[];
    gm=[];
    Hm=[];
end

