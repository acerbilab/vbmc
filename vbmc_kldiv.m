function kldivs = vbmc_kldiv(vp1,vp2,Ns,gauss_approximation,origflag)
%VBMC_KLDIV Compute KL divergences between two variational posteriors.

if nargin < 3 || isempty(Ns); Ns = 1e5; end
if nargin < 4 || isempty(gauss_approximation); gauss_approximation = false; end
if nargin < 5 || isempty(origflag); origflag = true; end

kldivs = NaN(1,2);

D = vp1.D;           % Number of dimensions

%try
    if gauss_approximation
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

        [kldivs(1),kldivs(2)] = mvnkl(q1mu,q1sigma,q2mu,q2sigma);
        
    else
        MINP = realmin;
        
        xx1 = vbmc_rnd(Ns,vp1,origflag,1);        
        q1 = vbmc_pdf(vp1,xx1,origflag);
        q2 = vbmc_pdf(vp2,xx1,origflag);
        q1(q1 == 0 | ~isfinite(q1)) = 1;    % Ignore these points
        q2(q2 == 0 | ~isfinite(q2)) = MINP;
        kldivs(1) = -mean(log(q2) - log(q1));

        xx2 = vbmc_rnd(Ns,vp2,origflag,1);
        q1 = vbmc_pdf(vp1,xx2,origflag);
        q2 = vbmc_pdf(vp2,xx2,origflag);
        q1(q1 == 0 | ~isfinite(q1)) = MINP;
        q2(q2 == 0 | ~isfinite(q2)) = 1;    % Ignore these points
        kldivs(2) = -mean(log(q1) - log(q2));
        
    end
    
    kldivs = max(kldivs,0); % Correct for numerical errors
    
%catch
    
    % Could not compute KL divs
    
%end