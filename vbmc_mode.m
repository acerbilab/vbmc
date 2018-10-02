function [x,fval] = vbmc_mode(vp,origflag)
%VBMC_MODE Find mode of VBMC posterior approximation.

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

% Get mode and its pdf value
x = xmin(idx,:);
fval = exp(-fval);

    function [y,dy] = nlnpdf(x)
    %NLNPDF Negative log posterior pdf and its gradient.
        if nargout > 1
            [y,dy] = vbmc_pdf(x,vp,origflag,1);
            y = -y; dy = -dy;            
        else
            y = -vbmc_pdf(x,vp,origflag,1);
        end
    end
end