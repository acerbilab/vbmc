function varargout = warpvars(varargin)
%WARPVARS Linear and nonlinear transformation of variables.
%
%  TRINFO = PDFTRANS(NVARS,LB,UB) returns the transformation structure 
%  TRINFO for a problem with NVARS dimensions and lower/upper bounds
%  respectively LB and UB. LB and UB are either scalars or row arrays that 
%  can contain real numbers and Inf's
%  The ordering LB <= UB needs to hold coordinate-wise.
%
%  Variables with lower or upper bounds are transformed via a log transform.
%  Variables with both lower and upper bounds are transformed via a logit
%  transform. 
%
%  Y = TRANSVARS(X,'dir',TRINFO) performs direct transform of constrained 
%  variables X into unconstrained variables Y according to transformation 
%  encoded in structure TRINFO. X must be a N x NVARS array, where N is the 
%  number of input data and NVARS is the number of dimensions.
%
%  X = TRANSVARS(Y,'inv',TRINFO) performs inverse transform of unconstrained 
%  variables Y into constrained variables X.
%
%  P = TRANSVARS(Y,'prob',TRINFO) returns probability multiplier for the 
%  original pdf evaluated at f^{-1}(Y).
%
%  LP = TRANSVARS(Y,'logprob',TRINFO) returns log probability term for the 
%  original log pdf evaluated at f^{-1}(Y).

%  Author: Luigi Acerbi
%  e-mail: luigi.acerbi@gmail.com

if nargin < 3
    error('TRANSVARS requires a minimum of three input arguments.');
end

%% Transform variables
if isstruct(varargin{3})

    trinfo = varargin{3};    

    if isempty(trinfo)
        varargout{1} = varargin{1}; % Return untransformed input
    else
        
        action = varargin{2};
        if isempty(action)
            error('The transformation direction cannot be empty. Allowed values are direct (''dir'' or ''d'') and inverse (''inv'' or ''i'').');
        end
        
        scale = [];
        if isfield(trinfo,'scale') && ~isempty(trinfo.scale) && any(trinfo.scale ~= 1)
            scale = trinfo.scale;
        end

        if ~isfield(trinfo,'R_mat'); trinfo.R_mat = []; end
        
        nvars = numel(trinfo.lb_orig);  % Number of variables
        
        switch lower(action(1))
        %% DIRECT TRANSFORM
            case 'd'    % Direct transform
                x = varargin{1};            
                y = x;
                a = trinfo.lb_orig;
                b = trinfo.ub_orig;
                mu = trinfo.mu;
                delta = trinfo.delta;
                
                % Unbounded scalars (possibly center and rescale)
                idx = trinfo.type == 0;
                if any(idx)
                    y(:,idx) = bsxfun(@rdivide,bsxfun(@minus,x(:,idx),mu(idx)),delta(idx));
                end

                % Lower bounded scalars
                idx = trinfo.type == 1;
                if any(idx)
                    y(:,idx) = log(bsxfun(@minus, x(:,idx), a(idx)));
                end

                % Upper bounded scalars
                idx = trinfo.type == 2;
                if any(idx)
                    y(:,idx) = log(bsxfun(@minus, b(idx), x(:,idx)));
                end

                % Lower and upper bounded scalars
                idx = trinfo.type == 3;
                if any(idx)
                    z = bsxfun(@rdivide, bsxfun(@minus, x(:,idx), a(idx)), ...
                        b(idx) - a(idx)); 
                    y(:,idx) = log(z./(1-z));
                    y(:,idx) = bsxfun(@rdivide,bsxfun(@minus,y(:,idx),mu(idx)),delta(idx));
                end
                
                % Lower and upper bounded scalars with Beta CDF transform
                idx = trinfo.type == 4;
                if any(idx)
                    alpha = trinfo.alpha;
                    beta = trinfo.beta;
                    for ii = find(idx)
                        % z = betacdf((x(:,ii) - a(ii)) / (b(ii) - a(ii)),alpha(ii),beta(ii));
                        z = min(max(eps,betacdf((x(:,ii) - a(ii)) / (b(ii) - a(ii)),alpha(ii),beta(ii))),1-eps);
                        y(:,ii) = log(z./(1-z));
                    end
                end

                % Lower and upper bounded scalars with Kumaraswamy CDF transform
                idx = trinfo.type == 5;
                if any(idx)
                    alpha = trinfo.alpha;
                    beta = trinfo.beta;
                    for ii = find(idx)
                        % z = kumarcdf((x(:,ii) - a(ii)) / (b(ii) - a(ii)),alpha(ii),beta(ii));
                        % p = min(max(eps,kumarcdf(z,alpha(ii),beta(ii))),1-eps);
                        % y(:,ii) = log(p./(1-p));
                        z = (x(:,ii) - a(ii)) / (b(ii) - a(ii));
                        % p = 1 - (1 - z.^alpha(ii)).^beta(ii);
                        y(:,ii) = log1p(-(1 - z.^alpha(ii)).^beta(ii)) - beta(ii)*log1p(-z.^alpha(ii));
                    end
                end

                % Lower and upper bounded scalars with Kumaraswamy-logistic-power transform
                idx = trinfo.type == 6;
                if any(idx)
                    alpha = trinfo.alpha;
                    beta = trinfo.beta;
                    mu = trinfo.mu;
                    gamma = trinfo.gamma;
                    for ii = find(idx)
                        z = (x(:,ii) - a(ii)) / (b(ii) - a(ii));
                        y(:,ii) = log1p(-(1 - z.^alpha(ii)).^beta(ii)) - beta(ii)*log1p(-z.^alpha(ii)) - mu(ii);
                        y(:,ii) = sign(y(:,ii)).*abs(y(:,ii)).^gamma(ii);
                    end
                end
                
                % Lower and upper bounded scalars with nonparametric CDF transform
                idx = trinfo.type == 7;
                if any(idx)
                    xspace = trinfo.xspace;
                    pspace = trinfo.pspace;
                    for ii = find(idx)
                        y(:,ii) = norminv(interp1(xspace(ii,:),pspace(ii,:),x(:,ii)));
                    end
                end

                % Lower and upper bounded scalars with GMM CDF transform
                idx = trinfo.type == 8;
                if any(idx)
                    for ii = find(idx)
                        gmm = trinfo.gmm{ii};
                        z = (x(:,ii) - a(ii)) / (b(ii) - a(ii));
                        z = gmm.lambda*z + (1-gmm.lambda)*(gmm1cdf(z,gmm.w,gmm.Mu,gmm.Sigma)-gmm.Min)./gmm.Norm;
                        y(:,ii) = norminv(z);
                        % y(:,ii) = logiinv(z);
                    end
                end
                
                % Lower and upper bounded scalars with Kumaraswamy-logit transform
                idx = trinfo.type == 9;
                if any(idx)
                    alpha = trinfo.alpha;
                    beta = trinfo.beta;
                    for ii = find(idx)
                        z = (x(:,ii) - a(ii)) / (b(ii) - a(ii));
                        y(:,ii) = log1p(-(1 - z.^alpha(ii)).^beta(ii)) - beta(ii)*log1p(-z.^alpha(ii));
                    end
                end                
                
                % Unbounded with logistic-Kumaraswamy-logit transform
                idx = trinfo.type == 10;
                if any(idx)
                    alpha = trinfo.alpha;
                    beta = trinfo.beta;
                    for ii = find(idx)
                        z = (x(:,ii)-mu(ii)) / delta(ii);
                        z = exp(z)./(exp(z)+1);
                        y(:,ii) = log1p(-(1 - z.^alpha(ii)).^beta(ii)) - beta(ii)*log1p(-z.^alpha(ii));
                    end
                end
                                
                % Rotate output
                if ~isempty(trinfo.R_mat); y = y*trinfo.R_mat; end
                
                % Rescale output
                if ~isempty(scale); y = bsxfun(@rdivide,y,scale); end
                
                varargout{1} = y;
                
            %% INVERSE TRANSFORM
            case 'i'    % Inverse transform
                y = varargin{1};                
                % Rescale input
                if ~isempty(scale); y = bsxfun(@times,y,scale); end
                
                % Rotate input
                if ~isempty(trinfo.R_mat); y = y*trinfo.R_mat'; end        
                                
                x = y;
                a = trinfo.lb_orig;
                b = trinfo.ub_orig;
                mu = trinfo.mu;
                delta = trinfo.delta;                

                % Unbounded scalars (possibly unscale and uncenter)
                idx = trinfo.type == 0;
                if any(idx)
                    x(:,idx) = bsxfun(@plus,bsxfun(@times,y(:,idx),delta(idx)),mu(idx));
                end
                
                % Lower bounded scalars
                idx = trinfo.type == 1;
                if any(idx)
                    x(:,idx) = bsxfun(@plus, exp(y(:,idx)), a(idx));
                end

                % Upper bounded scalars
                idx = trinfo.type == 2;
                if any(idx)
                    x(:,idx) = bsxfun(@minus, b(idx), exp(y(:,idx)));
                end

                % Lower and upper bounded scalars
                idx = trinfo.type == 3;
                if any(idx)
                    x(:,idx) = bsxfun(@plus,bsxfun(@times,y(:,idx),delta(idx)),mu(idx));
                    x(:,idx) = bsxfun(@plus, a(:,idx), bsxfun(@times, ...
                        b(idx)-a(idx), 1./(1+exp(-x(:,idx)))));
                end
                
                % Lower and upper bounded scalars with Beta CDF transform
                idx = trinfo.type == 4;
                if any(idx)
                    alpha = trinfo.alpha;
                    beta = trinfo.beta;
                    for ii = find(idx)
                        z = 1./(1+exp(-y(:,ii)));
                        x(:,ii) = a(ii) + (b(ii)-a(ii))*betainv(z,alpha(ii),beta(ii));
                    end
                end                

                % Lower and upper bounded scalars with Kumaraswamy CDF transform
                idx = trinfo.type == 5;
                if any(idx)
                    alpha = trinfo.alpha;
                    beta = trinfo.beta;
                    for ii = find(idx)
                        % z = 1./(1+exp(-y(:,ii)));
                        % x(:,ii) = a(ii) + (b(ii)-a(ii))*kumarinv(z,alpha(ii),beta(ii));
                        z = exp(-y(:,ii))./(1+exp(-y(:,ii)));
                        x(:,ii) = a(ii) + (b(ii)-a(ii))*(1-z.^(1/beta(ii))).^(1/alpha(ii));
                    end
                end
                
                % Lower and upper bounded scalars with Kumaraswamy-logistic-power transform
                idx = trinfo.type == 6;
                if any(idx)
                    alpha = trinfo.alpha;
                    beta = trinfo.beta;
                    mu = trinfo.mu;
                    gamma = trinfo.gamma;                    
                    for ii = find(idx)
                        z = sign(y(:,ii)).*abs(y(:,ii)).^(1/gamma(ii)) + mu(ii);
                        z = exp(-z)./(1+exp(-z));
                        x(:,ii) = a(ii) + (b(ii)-a(ii))*(1-z.^(1/beta(ii))).^(1/alpha(ii));
                    end
                end
                
                % Lower and upper bounded scalars with nonparametric CDF transform
                idx = trinfo.type == 7;
                if any(idx)
                    xspace = trinfo.xspace;
                    pspace = trinfo.pspace;
                    for ii = find(idx)
                        x(:,ii) = interp1(pspace(ii,:),xspace(ii,:),normcdf(y(:,ii)));
                    end
                end

                % Lower and upper bounded scalars with GMM CDF transform
                idx = trinfo.type == 8;
                if any(idx)                    
                    for ii = find(idx)
                        gmm = trinfo.gmm{ii};
                        z = normcdf(y(:,ii));
                        % z = logicdf(y(:,ii));
                        for j = 1:size(z,1)
                            z(j) = tgmminv(z(j),gmm);
                        end
                        x(:,ii) = a(ii) + (b(ii)-a(ii)).*z;
                    end
                end
                
                % Lower and upper bounded scalars with Kumaraswamy-logit transform
                idx = trinfo.type == 9;
                if any(idx)
                    alpha = trinfo.alpha;
                    beta = trinfo.beta;
                    for ii = find(idx)
                        z = exp(-y(:,ii))./(1+exp(-y(:,ii)));   % 1 - logistic(z)
                        x(:,ii) = a(ii) + (b(ii)-a(ii))*(1-z.^(1/beta(ii))).^(1/alpha(ii));
                    end
                end
                
                % Unbounded scalars with logistic-Kumaraswamy-logit transform
                idx = trinfo.type == 10;
                if any(idx)
                    alpha = trinfo.alpha;
                    beta = trinfo.beta;                                        
                    for ii = find(idx)
                        z = exp(-y(:,ii))./(1+exp(-y(:,ii)));   % 1 - logistic(z)
                        % z = (1-z.^(1/beta(ii))).^(1/alpha(ii));
                        z = 1/alpha(ii)*log1p(-z.^(1/beta(ii))) - log1p(-(1-z.^(1/beta(ii))).^(1/alpha(ii)));
                        x(:,ii) = z*delta(ii) + mu(ii);
                    end
                end
                                                
                % Force to stay within bounds
                a(isfinite(a)) = a(isfinite(a)) + eps(a(isfinite(a)));
                b(isfinite(b)) = b(isfinite(b)) - eps(b(isfinite(b)));
                x = bsxfun(@min,bsxfun(@max,x,a),b);
                varargout{1} = x;
                
            %% PDF (OR LOG PDF) CORRECTION           
            case {'p','l','g'}  % pdf (or log pdf) correction
                y = varargin{1};
                % Rescale input
                if ~isempty(scale); y = bsxfun(@times,y,scale); end

                % Rotate input
                if ~isempty(trinfo.R_mat); y = y*trinfo.R_mat'; end        
                
                logpdf_flag = strcmpi(action(1),'l');
                if logpdf_flag
                    p = zeros(size(y,1),nvars);
                else
                    p = ones(size(y,1),nvars);
                end
                grad_flag = strcmpi(action(1),'g');
                
                a = trinfo.lb_orig;
                b = trinfo.ub_orig;
                mu = trinfo.mu;
                delta = trinfo.delta;                
                
                % Unbounded scalars
                idx = trinfo.type == 0;
                if any(idx)
                    p(:,idx) = repmat(log(delta(idx)),[size(p,1),1]);
                end
                                
                % Lower or upper bounded scalars
                idx = trinfo.type == 1 | trinfo.type == 2;
                if any(idx)
                    p(:,idx) = y(:,idx);
                end

                % Lower and upper bounded scalars
                idx = trinfo.type == 3;
                if any(idx)
                    y(:,idx) = bsxfun(@plus,bsxfun(@times,y(:,idx),delta(idx)),mu(idx));
                    z = -log1p(exp(-y(:,idx)));
                    p(:,idx) = bsxfun(@plus, log(b(idx)-a(idx)), -y(:,idx) + 2*z);
                    p(:,idx) = bsxfun(@plus, p(:,idx), log(delta(idx)));
                end
                
                % Lower and upper bounded scalars with Beta CDF transform
                idx = trinfo.type == 4;
                if any(idx)
                    for ii = find(idx)
                        alpha = trinfo.alpha;
                        beta = trinfo.beta;

                        z = -log1p(exp(-y(:,ii)));
                        x = min(max(eps,betainv(1./(1+exp(-y(:,ii))),alpha(ii),beta(ii))),1-eps);
                        logbeta = (alpha(ii)-1)*log(x) + (beta(ii)-1)*log1p(-x) ...
                            + gammaln(alpha(ii)+beta(ii)) - gammaln(alpha(ii)) - gammaln(beta(ii));
                        p(:,ii) = log(b(ii)-a(ii)) -logbeta -y(:,ii) + 2*z;

                        if any(~isfinite(p))
                            fprintf('aaaa!');
                        end
                    end
                end
                
                % Lower and upper bounded scalars with Kumaraswamy CDF transform
                idx = trinfo.type == 5;
                if any(idx)
                    for ii = find(idx)
                        alpha = trinfo.alpha;
                        beta = trinfo.beta;

                        z = -log1p(exp(-y(:,ii)));
                        % x = kumarinv(1./(1+exp(-y(:,ii))),alpha(ii),beta(ii));

                        %p = 1./(1+exp(-y(:,ii)));
                        %x = (-((-(p-1)).^(1/beta(ii))-1)).^(1/alpha(ii));
                        %log(-(p-1)) = -y(,::) - log1p(exp(-y(:,ii)))); 


                        %(1/beta(ii)).*log(-(p-1))

                        % x = min(max(eps,kumarinv(1./(1+exp(-y(:,ii))),alpha(ii),beta(ii))),1-eps);
                        u = 1./(1 + exp(-y(:,ii)));
                        logf = (1-1/alpha(ii))*log1p(-(1-u).^(1/beta(ii))) + (1-1/beta(ii))*(-y(:,ii)+z) ...
                            + log(alpha(ii)) + log(beta(ii));

                        % Special case for very small u
                        %idx_small = u < 1e12;
                        %logf(idx_small) = (1-1/alpha(ii))*(-y(idx_small,ii) + z - log(beta(ii))) + (1-1/beta(ii))*(-y(idx_small,ii)+z(idx_small)) ...
                        %    + log(alpha(ii)) + log(beta(ii));

                        %logf = (alpha(ii)-1)*log(x) + (beta(ii)-1)*log1p(-x.^alpha(ii)) ...
                        %    + log(alpha(ii)) + log(beta(ii));
                        p(:,ii) = log(b(ii)-a(ii)) -logf -y(:,ii) + 2*z;

                        if any(~isfinite(p))
                            p(~isfinite(p)) = -Inf;
                            fprintf('aaaa!');
                        end
                    end
                end
                
                % Lower and upper bounded scalars with Kumaraswamy-logistic-power transform
                idx = trinfo.type == 6;
                if any(idx)
                    for ii = find(idx)
                        alpha = trinfo.alpha;
                        beta = trinfo.beta;
                        mu = trinfo.mu;
                        gamma = trinfo.gamma;                        
                        
                        yl = sign(y(:,ii)).*abs(y(:,ii)).^(1/gamma(ii)) + mu(ii);

                        z = -log1p(exp(-yl));
                        u = 1./(1 + exp(-yl));
                        logf = (1-1/alpha(ii))*log1p(-(1-u).^(1/beta(ii))) + (1-1/beta(ii))*(-yl+z) ...
                            + log(alpha(ii)) + log(beta(ii));
                        logf = logf + log(gamma(ii)) + (gamma(ii)-1)*log(abs(yl-mu(ii)));
                        p(:,ii) = log(b(ii)-a(ii)) -logf -yl + 2*z;
                    end
                end
                
                % Lower and upper bounded scalars with nonparametric CDF transform
                idx = trinfo.type == 7;
                if any(idx)
                    xspace = trinfo.xspace;
                    pspace = trinfo.pspace;
                    for ii = find(idx)
                        z = -0.5*log(2*pi) -0.5*y(:,ii).^2;

                        yinv = normcdf(y(:,ii));
                        [~,pos] = histc(yinv,pspace(ii,:));
                        dx = [-log(diff(xspace(ii,:))),Inf];                            
                        logf = dx(pos);

                        p(:,ii) = log(b(ii)-a(ii)) -logf(:) + z;
                    end
                end
                
                % Lower and upper bounded scalars with GMM CDF transform
                idx = trinfo.type == 8;
                if any(idx)
                    for ii = find(idx)
                        gmm = trinfo.gmm{ii};

                        z = -0.5*log(2*pi) -0.5*y(:,ii).^2;
                        % z = -y(:,ii) - 2*log1p(exp(-y(:,ii)));

                        yinv = normcdf(y(:,ii));
                        for j = 1:size(yinv,1); yinv(j) = tgmminv(yinv(j),gmm); end

                        logf = log(gmm.lambda + (1-gmm.lambda)*gmm1pdf(yinv,gmm.w,gmm.Mu,gmm.Sigma)./gmm.Norm);

                        p(ii) = log(b(ii)-a(ii)) - logf + z;

                        if any(~isfinite(p))
                            p(~isfinite(p)) = -Inf;
                            fprintf('aaaa!');
                        end
                    end
                end
                
                % Lower and upper bounded scalars with Kumaraswamy-logit transform
                idx = trinfo.type == 9;
                if any(idx)
                    for ii = find(idx)
                        alpha = trinfo.alpha;
                        beta = trinfo.beta;                        
                        nf = (b(ii)-a(ii))/alpha(ii)/beta(ii);
                        k = 1./(1+exp(-y(:,ii)));
                        
                        logk = -log1p(exp(-y(:,ii)));
                        log1mk = logk - y(:,ii);
                        %z = (1-(1-k).^(1/beta(ii))).^(1/alpha(ii));
                        % 1 - z^alpha = (1-k).^(1/beta(ii))                            
                        logz = 1/alpha(ii)*log1p(-(1-k).^(1/beta(ii)));
                        p(:,ii) = log(nf) + (1/beta(ii)-1) * log1mk + (1-alpha(ii))*logz -y(:,ii)+2*logk;
                    end
                end
                
                % Unbounded scalars with logistic-Kumaraswamy-logit transform
                idx = trinfo.type == 10;
                if any(idx)
                    for ii = find(idx)
                        alpha = trinfo.alpha;
                        beta = trinfo.beta;                        
                        nf = delta(ii)/alpha(ii)/beta(ii);
                        
                        k = 1./(1+exp(y(:,ii)));
                        z = (1-k.^(1/beta(ii))).^(1/alpha(ii));                        
                        logk = -log1p(exp(y(:,ii)));
                        p(:,ii) = log(nf) + y(:,ii) + (1+1/beta(ii)).*logk - log1p(-k.^(1/beta(ii))) - log1p(-z);
                    end
                end
                

                %if ~isempty(trinfo.R_mat) && lower(action(1)) == 'g'
                %    p = p*(trinfo.R_mat*diag();
                %end
                
                % Scale transform
                if ~isempty(scale) && ~grad_flag
                    p = bsxfun(@plus,p,log(scale));
                end
                
                if ~grad_flag; p = sum(p,2); end
                if ~logpdf_flag; p = exp(p); end
                
                varargout{1} = p;
                
            %% FIRST DERIVATIVE WRT TRANSFORMATION PARAMETERS (ignores final rotation and scaling)
            case {'f'} 
                y = varargin{1};
                nvars = numel(trinfo.lb_orig);
                
                % Rescale input
                if ~isempty(scale); y = bsxfun(@times,y,scale); end

                % Rotate input
                if ~isempty(trinfo.R_mat); y = y*trinfo.R_mat'; end        
                
                a = trinfo.lb_orig;
                b = trinfo.ub_orig;
                delta = trinfo.delta;                

                % Lower and upper bounded scalars with Kumaraswamy-logit transform
                % and unbounded scalars with logistic-Kumaraswamy-logit transform
                idx = (trinfo.type == 9 | trinfo.type == 10);
                if any(idx)
                    for ii = find(idx)
                        alpha = trinfo.alpha;
                        beta = trinfo.beta;                        

                        k = 1./(1+exp(-y(:,ii)));                        
                        talpha = 1 - (1-k).^(1/beta(ii));                        
                        logt = 1/alpha(ii) .* log1p(-(1-k).^(1/beta(ii)));
                                                
                        p(:,ii) = talpha.*beta(ii).*logt./(1-talpha)./k;
                        p(:,ii+nvars) = -log1p(-talpha)./k;
                    end
                end
                
                varargout{1} = p;
                
            %% MIXED DERIVATIVE WRT TRANSFORMATION PARAMETERS of FIRST DERIVATIVE
            % (ignores final rotation and scaling)          
            case {'m'}
                y = varargin{1};
                nvars = numel(trinfo.lb_orig);
                
                % Rescale input
                if ~isempty(scale); y = bsxfun(@times,y,scale); end

                % Rotate input
                if ~isempty(trinfo.R_mat); y = y*trinfo.R_mat'; end        
                
                a = trinfo.lb_orig;
                b = trinfo.ub_orig;
                delta = trinfo.delta;                

                % Lower and upper bounded scalars with Kumaraswamy-logit transform
                % and unbounded scalars with logistic-Kumaraswamy-logit transform
                idx = (trinfo.type == 9 | trinfo.type == 10);
                if any(idx)
                    for ii = find(idx)
                        alpha = trinfo.alpha;
                        beta = trinfo.beta;                        

                        k = 1./(1+exp(-y(:,ii)));                        
                        talpha = 1 - (1-k).^(1/beta(ii));                        
                        logt = 1/alpha(ii) .* log1p(-(1-k).^(1/beta(ii)));
                        t = talpha.^(1/alpha(ii));
                        
                        if trinfo.type(ii) == 9
                            nf = - 1./(b(ii) - a(ii)) ./ t;
                        else
                            nf = (t-1)./delta(ii);                       
                        end                        
                        den = 1 ./ k.^2 ./ (talpha-1) .* nf;                                                
                        p(:,ii) = talpha.*beta(ii).*((-k.*(1+alpha(ii).*logt)) + talpha.*(1 + (1-k).*(-1+alpha(ii).*beta(ii).*logt))) ...
                            .* den ./ (talpha - 1);
                        p(:,ii+nvars) = talpha .* alpha(ii) .* (1 + (1-k).*(-1 - y(:,ii) - log1p(exp(-y(:,ii))))) ...
                            .* den;
                    end
                end
                
                varargout{1} = p;
                
            otherwise
                error(['Unkwnown transformation direction ''' action '''. Allowed values are direct (''dir'' or ''d'') and inverse (''inv'' or ''i'').']);
        end
    end
    
else
%% Create transform

    nvars = varargin{1};
    lb = varargin{2}(:)';
    ub = varargin{3}(:)';
    if nargin > 3
        plb = varargin{4}(:)';
        pub = varargin{5}(:)';
    else
        plb = []; pub = [];
    end
            
    % Empty LB and UB are Infs
    if isempty(lb); lb = -Inf; end
    if isempty(ub); ub = Inf; end

    % Empty plausible bounds equal hard bounds
    if isempty(plb); plb = lb; end
    if isempty(pub); pub = ub; end
    
    % Convert scalar inputs to row vectors
    if isscalar(lb); lb = lb*ones(1,nvars); end
    if isscalar(ub); ub = ub*ones(1,nvars); end
    if isscalar(plb); plb = plb*ones(1,nvars); end
    if isscalar(pub); pub = pub*ones(1,nvars); end
    
    % Check that the order of bounds is respected
    assert(all(lb <= plb & plb < pub & pub <= ub), ...
        'Variable bounds should be LB <= PLB < PUB <= UB for all variables.');
    
    % Transform to log coordinates
    trinfo.lb_orig = lb;
    trinfo.ub_orig = ub;
    
    trinfo.type = zeros(1,nvars);    
    for i = 1:nvars
        if isfinite(lb(i)) && isinf(ub(i)); trinfo.type(i) = 1; end
        if isinf(lb(i)) && isfinite(ub(i)); trinfo.type(i) = 2; end
        if isfinite(lb(i)) && isfinite(ub(i)) && lb(i) < ub(i); trinfo.type(i) = 3; end
    end
    
    % Centering (used only for unbounded variables)
    trinfo.mu = zeros(1,nvars);
    trinfo.delta = ones(1,nvars);
    for i = 1:nvars
        if isfinite(plb(i)) && isfinite(pub(i))
            trinfo.mu(i) = 0.5*(plb(i)+pub(i));
            trinfo.delta(i) = pub(i)-plb(i);
        end
    end
        
    varargout{1} = trinfo;
    
end

end

%--------------------------------------------------------------------------
function y = tgmminv(p,gmm)
    if p <= 0; p = eps; elseif p >= 1; p = 1 - eps; end    % Correct bounds
    fun = @(z) tgmmfzero(z,p,gmm.w,gmm.Mu,gmm.Sigma,gmm.Min,gmm.iMax,gmm.Norm,gmm.lambda);
    y = qfzero(fun,[-eps,1+eps]);
    if y <= 0; y = eps; elseif y >= 1; y = 1 - eps; end    % Correct bounds
end