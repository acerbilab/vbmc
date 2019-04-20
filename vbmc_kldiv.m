function kls = vbmc_kldiv(vp1,vp2,Ns,gaussflag)
%VBMC_KLDIV Kullback-Leibler divergence between two variational posteriors.
%   KLS = VBMC_KLDIV(VP1,VP2) returns an estimate of the (asymmetric) 
%   Kullback-Leibler (KL) divergence between two variational posterior 
%   distributions VP1 and VP2. KLS is a 2-element vector whose first element
%   is KL(VP1||VP2) and the second element is KL(VP2||VP1). The symmetrized
%   KL divergence can be computed as mean(KLS).
%
%   KLS = VBMC_KLDIV(VP1,VP2,NS) uses NS random draws to estimate each
%   KL divergence (default NS=1e5).
%
%   KLS = VBMC_KLDIV(VP1,VP2,NS,GAUSSFLAG) computes the "Gaussianized" 
%   KL-divergence if GAUSSFLAG=1, that is the KL divergence between two
%   multivariate normal distibutions with the same moments as the variational
%   posteriors given as inputs. Otherwise, the standard KL-divergence is 
%   returned for GAUSSFLAG=0 (default).
%
%   See also VBMC, VBMC_PDF, VBMC_RND, VBMC_DIAGNOSTICS.

if nargin < 3 || isempty(Ns); Ns = 1e5; end
if nargin < 4 || isempty(gaussflag); gaussflag = false; end

% This was removed because the comparison *has* to be in original space,
% given that the transform might change for distinct variational posteriors
% if nargin < 5 || isempty(origflag); origflag = true; end
origflag = true;

kls = NaN(1,2);

D = vp1.D;           % Number of dimensions

%try
    if gaussflag
        if Ns == 0  % Analytical calculation
            if origflag
                error('vbmc_kldiv:NoAnalyticalMoments', ...
                    'Analytical moments are available only for the transformed space.')
            end
            [q1mu,q1sigma] = vbmc_moments(vp1,0);
            [q2mu,q2sigma] = vbmc_moments(vp2,0);
        else        % Numerical moments
            [q1mu,q1sigma] = vbmc_moments(vp1,origflag,Ns);
            [q2mu,q2sigma] = vbmc_moments(vp2,origflag,Ns);
        end

        [kls(1),kls(2)] = mvnkl(q1mu,q1sigma,q2mu,q2sigma);
        
    else
        MINP = realmin;
        
        xx1 = vbmc_rnd(vp1,Ns,origflag,1);        
        q1 = vbmc_pdf(vp1,xx1,origflag);
        q2 = vbmc_pdf(vp2,xx1,origflag);
        q1(q1 == 0 | ~isfinite(q1)) = 1;    % Ignore these points
        q2(q2 == 0 | ~isfinite(q2)) = MINP;
        kls(1) = -mean(log(q2) - log(q1));

        xx2 = vbmc_rnd(vp2,Ns,origflag,1);
        q1 = vbmc_pdf(vp1,xx2,origflag);
        q2 = vbmc_pdf(vp2,xx2,origflag);
        q1(q1 == 0 | ~isfinite(q1)) = MINP;
        q2(q2 == 0 | ~isfinite(q2)) = 1;    % Ignore these points
        kls(2) = -mean(log(q1) - log(q2));
        
    end
    
    kls = max(kls,0); % Correct for numerical errors
    
%catch
    
    % Could not compute KL divs
    
%end