function [m,C,gm,gC,Hm,HC] = posterior_gp(XStar, gp, sample,flag)
% can construct cells of flags if desired e.g.
% {'var_not_cov','noise_corrected_variance'}

XData = gp.X_data;

if nargin<3
    sample=1;
end
if nargin<4
    flag = '';
end

% attempt to adjust our predictions for the deleterious effects of added
% jitter
jitter_corrected = nargout~=1 && ...
    any(strcmpi(flag,'jitter_corrected'));

% include the extra noise in our predictive variance to reflect the
% fact that we make predicions about the noisy measurements, rather
% than the latent variable
noise_corrected_variance = nargout~=1 && ...
    any(strcmpi(flag,'noise_corrected_variance'));
if noise_corrected_variance
    Noise = get_noise(gp, 'plain');
end

% actually return a vector of variances in the place of a covariance
% matrix
varnotcov=any(strcmpi(flag,'var_not_cov'));

% If nocov, we do not compute the covariance
nocov = nargout==1 || any(strcmpi(flag,'no_cov'));

% You can work this one out yourself
nomean = any(strcmpi(flag,'no_mean'));

hs = gp.hypersamples(sample).hyperparameters;
K = gp.covfn('plain');
Mu = get_mu(gp, 'plain');
if nargout>2
    %DK will always assume derivative is taken wrt first argument
    DK = gp.covfn('grad inputs');
    DMu = get_mu(gp, 'grad inputs');
	if nargout>4
        DDK = gp.covfn('hessian inputs');
        DDMu = get_mu(gp, 'hessian inputs');
    end
end
if varnotcov
    vecK = gp.covfn('vector');
end

NData=size(XData,1);
NStar=size(XStar,1);

%gp = gpparams(XData,YData,gp,sample,'refresh');



if isfield(gp.hypersamples(sample),'cholK')
    cholK=gp.hypersamples(sample).cholK;
    datatwothirds=gp.hypersamples(sample).datatwothirds;
    if jitter_corrected
        K_data=gp.hypersamples(sample).K;
        jitters=gp.hypersamples(sample).jitters;
        yData = gp.y_data;
        datahalf=gp.hypersamples(sample).datahalf;
    end
else
    yData = gp.y_data;
    gp = revise_gp(XData, yData, gp, 'overwrite', [], sample);
    cholK=gp.hypersamples(sample).cholK;
    datatwothirds=gp.hypersamples(sample).datatwothirds;
end



lowr.UT=true;
lowr.TRANSA=true;
uppr.UT=true;

% DK(XStar,XData) & DDK(XStar,XData) are going to be cells, whose indices
% are over the different possible derivatives (there are NVariables in
% total). Each cell content is a matrix DK(XStar,XData) for a different
% possible derivative. This could be done using arrays but I find cells
% more intuitive.

MuStar = Mu(hs,XStar);
KStarData = K(hs, XStar, XData);

if jitter_corrected
    [K_data, cholK, datahalf, datatwothirds] = ...
        jitter_correction(jitters, KStarData, K_data, cholK, yData - Mu(hs, XData), datahalf, datatwothirds);
end

if ~nomean
    m = MuStar + KStarData*datatwothirds;     % Compute the objective function value at XStar
end

if nargout>1

    if varnotcov

        F = linsolve(cholK,KStarData',lowr);
        Kstst = vecK(hs, XStar, XStar);
        C = Kstst - sum(F.^2)'; 
    elseif ~nocov
        Kterm = linsolve(cholK,linsolve(cholK,KStarData',lowr),uppr);
        Kstst = K(hs, XStar,XStar);
        C = Kstst - KStarData*Kterm;
    end
    
    too_close = any(C<0);
    C(C<0) = eps;

    if noise_corrected_variance
        C = C + Noise(hs, XStar);
    end
 
    if nargout>2
        DMuStarData = DMu(hs, XStar);
        DKStarData = DK(hs, XStar, XData);

        % Gradient of the function evaluated at XStar
        %gm = DK(XStar,XData)*datatwothirds;  
        if ~nomean
            gm=cellfun(@(DMumat, DKmat) DMumat + DKmat*datatwothirds,...
                DMuStarData, DKStarData,...
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
                gC = cellfun(@(DKmat) -2*DKmat*Kterm, DKStarData,...
                    'UniformOutput',false);
            elseif too_close
                gC = cellfun(@(DKmat) 0*DKmat, DKStarData,...
                    'UniformOutput',false);
            end

            if nargout>4

                DDKStarData=DDK(hs, XStar, XData);
                DDMuStarData=DDMu(hs, XStar,XData);
                
                %Hm = DDK(XStar,XData)*datatwothirds;  % Hessian evaluated at XStar
                if ~nomean                   
                    Hm=cellfun(@(DDMuStarDatamat,DDKmat) ...
                        DDMuStarDatamat + DDKmat*datatwothirds,...
                        DDMuStarData,DDKStarData,'UniformOutput',false);
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

% hack to dodge problems with very small negative values
C = max(eps,C);

