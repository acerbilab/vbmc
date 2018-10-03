function [x,vp] = vbmc_mode(vp,origflag)
%VBMC_MODE Find mode of VBMC posterior approximation.
%   X = VBMC_PDF(VP) returns the mode of the variational posterior VP. 
%
%   X = VBMC_PDF(VP,ORIGFLAG) returns the mode of the variational posterior 
%   in the original parameter space if ORIGFLAG=1 (default), or in the 
%   transformed VBMC space if ORIGFLAG=0. The two modes are generally not 
%   equivalent, under a nonlinear transformation of variables.
%
%   [X,VP] = VBMC_PDF(...) returns the variational posterior with the mode 
%   stored in the VP struct.
%
%   See also VBMC, VBMC_MOMENTS, VBMC_PDF.

if nargin < 2 || isempty(origflag); origflag = true; end

if origflag && isfield(vp,'mode') && ~isempty(vp.mode)
    x = vp.mode;
else

    xmin = zeros(vp.K,vp.D);
    ff = Inf(vp.K,1);

    for k = 1:vp.K
        x0 = vp.mu(:,k)';
        if origflag; x0 = warpvars(x0,'inv',vp.trinfo); end

        if origflag
            opts = optimoptions('fmincon','GradObj','off','Display','off');

    %         mu = vp.mu';
    %         diam = (max(mu) - min(mu)) + 5*max(vp.sigma).*vp.lambda';
    %         LB_vp = warpvars(min(mu) - diam,'inv',vp.trinfo);
    %         UB_vp = warpvars(max(mu) + diam,'inv',vp.trinfo);     
    %         LB = max(vp.trinfo.lb_orig + sqrt(eps),LB_vp);
    %         UB = min(vp.trinfo.ub_orig - sqrt(eps),UB_vp);

             LB = vp.trinfo.lb_orig + sqrt(eps);
             UB = vp.trinfo.ub_orig - sqrt(eps);

            [xmin(k,:),ff(k)] = fmincon(@nlnpdf,x0,[],[],[],[],LB,UB,[],opts);        
        else
            opts = optimoptions('fminunc','GradObj','off','Display','off');
            [xmin(k,:),ff(k)] = fminunc(@nlnpdf,x0,opts);
        end
    end

    [fval,idx] = min(ff);

    % Get mode and store it
    x = xmin(idx,:);
    if nargout > 1 && origflag
        vp.mode = x;
    end
end

    function [y,dy] = nlnpdf(x)
    %NLNPDF Negative log posterior pdf and its gradient.
        if nargout > 1
            [y,dy] = vbmc_pdf(vp,x,origflag,1);
            y = -y; dy = -dy;            
        else
            y = -vbmc_pdf(vp,x,origflag,1);
        end
    end
end