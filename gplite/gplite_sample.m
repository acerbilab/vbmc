function [Xs,gp] = gplite_sample(gp,Ns,x0,method,logprior)
%GPLITE_SAMPLE Draw random samples from log pdf represented by GP.

if nargin < 3; x0 = []; end
if nargin < 4 || isempty(method); method = 'parallel'; end
if nargin < 5 || isempty(logprior); logprior = []; end

MaxBnd = 10;
D = size(gp.X,2);

widths = std(gp.X,[],1);
diam = max(gp.X) - min(gp.X);
LB = min(gp.X) - MaxBnd*diam;
UB = max(gp.X) + MaxBnd*diam;

% First, train GP
if ~isfield(gp,'post') || isempty(gp.post)
    % How many samples for the GP?
    if isfield(gp,'Ns') && ~isempty(gp.Ns)
        Ns_gp = gp.Ns;
    else
        Ns_gp = 0;
    end
    if isfield(gp,'Nopts') && ~isempty(gp.Nopts)
        options.Nopts = gp.Nopts;
    else
        options.Nopts = 1;  % Do only one optimization
    end
    gp = gplite_train(...
        [],Ns_gp,gp.X,gp.y,gp.meanfun,[],options);
end

% Recompute posterior auxiliary info if needed
if ~isfield(gp.post(1),'alpha') || isempty(gp.post(1).alpha)
    gp = gplite_post(gp);
end
    
if isempty(logprior)
    logpfun = @(x) gplite_pred(gp,x);
else
    logpfun = @(x) gplite_pred(gp,x) + logprior(x);    
end

switch method
    case {'slicesample','slicesamplebnd'}
        sampleopts.Burnin = ceil(Ns/10);
        sampleopts.Thin = 1;
        sampleopts.Display = 'off';
        sampleopts.Diagnostics = false;

        if isempty(x0)
            [~,idx0] = max(gp.y);
            x0 = gp.X(idx0,:);
        else
            x0 = x0(1,:);
        end
        Xs = slicesamplebnd(logpfun, ...
            x0,Ns,widths,LB,UB,sampleopts);
        
    case 'parallel'
        sampleopts.Burnin = ceil(Ns/5);
        sampleopts.Thin = 1;
        sampleopts.Display = 'off';
        sampleopts.Diagnostics = false;
        sampleopts.VarTransform = false;
        sampleopts.InversionSample = false;
        sampleopts.FitGMM = false;
        % sampleopts.TransitionOperators = {'transSliceSampleRD'};
        
        W = 2*(D+1);
        if isempty(x0)
            % Take starting points from high posterior density region
            hpd_frac = 0.25;
            N = numel(gp.y);
            N_hpd = min(N,max(W,round(hpd_frac*N)));
            if isempty(logprior)
                [~,ord] = sort(gp.y,'descend');
            else
                dy = logprior(gp.X);
                [~,ord] = sort(gp.y + dy,'descend');                
            end
            X_hpd = gp.X(ord(1:N_hpd),:);
            x0 = X_hpd(randperm(N_hpd,min(W,N_hpd)),:);
        end
        x0 = bsxfun(@min,bsxfun(@max,x0,LB),UB);
        Xs = eissample_lite(logpfun,x0,Ns,W,widths,LB,UB,sampleopts);
end

end