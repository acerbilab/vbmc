function [x,vp] = vbmc_mode(vp,origflag,nopts)
%VBMC_MODE Find mode of VBMC posterior approximation.
%   X = VBMC_PDF(VP) returns the mode of the variational posterior VP. 
%
%   X = VBMC_PDF(VP,ORIGFLAG) returns the mode of the variational posterior 
%   in the original parameter space if ORIGFLAG=1 (default), or in the 
%   transformed VBMC space if ORIGFLAG=0. The two modes are generally not 
%   equivalent, under a nonlinear transformation of variables.
%
%   X = VBMC_PDF(VP,ORIGFLAG,NOPTS) performs NOPTS optimizations from
%   different starting points to find the mode (by default, NOPTS is
%   the square root of the number of mixture components K, that is 
%   NOPTS = ceil(sqrt(K))).
%
%   [X,VP] = VBMC_PDF(...) returns the variational posterior with the mode 
%   stored in the VP struct.
%
%   See also VBMC, VBMC_MOMENTS, VBMC_PDF.

if nargin < 2 || isempty(origflag); origflag = true; end
if nargin < 3 || isempty(nopts); nopts = ceil(sqrt(vp.K)); end

nsamples = 1e5; % Samples for choosing starting points

if origflag && isfield(vp,'mode') && ~isempty(vp.mode)
    x = vp.mode;
else
    xmin = zeros(nopts,vp.D);
    ff = Inf(nopts,1);

    % Repeat optimization for NOPTS times
    for k = 1:nopts
        
        % Random initial set of points to choose starting point
        x0_mat = vbmc_rnd(vp,nsamples,origflag);
        
        % Add centers of components to initial set for first optimization
        if k == 1
            x0_mu = vp.mu';    
            if origflag
                x0_mu = warpvars_vbmc(x0_mu,'inv',vp.trinfo);
            end
            x0_mat = [x0_mat; x0_mu];
        end
        
        % Evaluate pdf at all points and start optimization from best
        y0_vec = nlnpdf(x0_mat);
        [~,idx] = min(y0_vec);        
        x0 = x0_mat(idx,:);

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

    % Get mode
    [fval,idx] = min(ff);
    x = xmin(idx,:);
    
    % Check old mode and store it if requested (only in original space)
    if origflag
        if isfield(vp,'mode') && ~isempty(vp.mode)
            oldnll = nlnpdf(vp.mode);
            if fval < oldnll
                if nargout > 1; vp.mode = x; end
            else
                x = vp.mode;
            end
        else
            if nargout > 1; vp.mode = x; end
        end
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