function [x,vp] = vbmc_mode(vp,origflag,nmax)
%VBMC_MODE Find mode of VBMC posterior approximation.
%   X = VBMC_PDF(VP) returns the mode of the variational posterior VP. 
%
%   X = VBMC_PDF(VP,ORIGFLAG) returns the mode of the variational posterior 
%   in the original parameter space if ORIGFLAG=1 (default), or in the 
%   transformed VBMC space if ORIGFLAG=0. The two modes are generally not 
%   equivalent, under a nonlinear transformation of variables.
%
%   X = VBMC_PDF(VP,ORIGFLAG,NMAX) performs up to NMAX optimizations from
%   different starting points to find the mode (by default, NMAX=50). The
%   optimizations are started from the centers of the mixture components in
%   the variational posterior (ordered in terms of posterior density), so 
%   the actual number of optimizations is min(K,NMAX), where K is the 
%   number of mixture components.
%
%   [X,VP] = VBMC_PDF(...) returns the variational posterior with the mode 
%   stored in the VP struct.
%
%   See also VBMC, VBMC_MOMENTS, VBMC_PDF.

if nargin < 2 || isempty(origflag); origflag = true; end
if nargin < 3 || isempty(nmax); nmax = 50; end

if origflag && isfield(vp,'mode') && ~isempty(vp.mode)
    x = vp.mode;
else    
    x0_mat = vp.mu';    
    if origflag
        x0_mat = warpvars_vbmc(x0_mat,'inv',vp.trinfo);
    end
    
    if nmax < vp.K
        y0_vec = nlnpdf(x0_mat);	% First, evaluate pdf at all modes        
        % Start from first NMAX solutions
        [~,ord] = sort(y0_vec,'ascend');
        x0_mat = x0_mat(ord(1:nmax),:);
    end
        
    xmin = zeros(size(x0_mat,1),vp.D);
    ff = Inf(size(x0_mat,1),1);

    for k = 1:size(x0_mat,1)
        x0 = x0_mat(k,:);

        if origflag
            opts = optimoptions('fmincon','GradObj','off','Display','off');
            LB = vp.trinfo.lb_orig + sqrt(eps);
            UB = vp.trinfo.ub_orig - sqrt(eps);
            x0 = min(max(x0,LB),UB);
            [xmin(k,:),ff(k)] = fmincon(@nlnpdf,x0,[],[],[],[],LB,UB,[],opts);        
        else
            opts = optimoptions('fminunc','GradObj','on','Display','off');
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