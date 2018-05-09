function [x,fval,gp] = gplite_fmin(gp,x0,maxflag)
%GPLITE_FMIN Find global minimum (or maximum) of GP.

if nargin < 2; x0 = 0; end
if nargin < 3 || isempty(maxflag); maxflag = 0; end

MaxBnd = 10;
hpd_frac = 0.5;
D = size(gp.X,2);
N0 = size(x0,1);
Nstarts = max(3,N0);

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
    options.Nopts = 1;  % Do only one optimization    
    gp = gplite_train(...
        [],Ns_gp,gp.X,gp.y,gp.meanfun,[],[],options);
end

% Start from the min (or max) of the training data
if maxflag
    [~,ord] = sort(gp.y,'descend');    
else
    [~,ord] = sort(gp.y,'ascend');
end

% Take best for sure
X = gp.X(ord,:);
x0 = [x0; X(1,:)];
X(1,:) = [];

if Nstarts > N0+1
    Nx = size(X,1);
    N_hpd = ceil(Nx*hpd_frac);
    idx = randperm(N_hpd,min(Nstarts-N0,N_hpd));
    x0 = [x0; X(idx,:)];
end

N0 = size(x0,1);
x = zeros(N0,D);
f = zeros(N0,1);
opts = optimoptions('fmincon','GradObj','off','Display','off');
for i = 1:N0
    [x(i,:),f(i)] = fmincon(@(x) optfun(x,gp,maxflag),x0(i,:),[],[],[],[],LB,UB,[],opts);
end

[fval,idx] = min(f);
x = x(idx,:);

if maxflag; fval = -fval; end

end

function [f,df] = optfun(x,gp,maxflag)

if nargout > 1
    [f,df] = gplite_pred(gp,x);
else
    f = gplite_pred(gp,x);
end

if maxflag  % Want to find maximum, swap sign
    f = -f;
    if nargout > 1; df = -df; end
end

end