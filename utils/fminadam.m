function [x,f,xtab,ftab] = fminadam(fun,x0,LB,UB,TolFun,MaxIter,master_stepsize)

if nargin < 3; LB = []; end
if nargin < 4; UB = []; end
if nargin < 5 || isempty(TolFun); TolFun = 0.001; end
if nargin < 6 || isempty(MaxIter); MaxIter = 1e4; end
if nargin < 7; master_stepsize = []; end

% Assign default parameters
master_stepsize_default.max = 0.1;
master_stepsize_default.min = 0.001;
master_stepsize_default.decay = 200;
for f = fields(master_stepsize_default)'
    if ~isfield(master_stepsize,f{:}) || isempty(master_stepsize.(f{:}))
        master_stepsize.(f{:}) = master_stepsize_default.(f{:});
    end
end

%% Adam with momentum
fudge_factor = sqrt(eps);
beta1 = 0.9;
beta2 = 0.999;
batchsize = 20;
TolX = 0.001;
TolX_max = 0.1;
TolFun_max = TolFun*100;

MinIter = batchsize*2;

nvars = numel(x0);
if isempty(LB); LB = -Inf(nvars,1); end
if isempty(UB); UB = Inf(nvars,1); end

m = 0; v = 0;
%xtab = zeros(nvars,batchsize*2);
xtab = zeros(nvars,MaxIter);

x = x0(:);
ftab = NaN(1,MaxIter);

for iter = 1:MaxIter
    idx = mod(iter-1,batchsize*2) + 1;
    isMinibatchEnd = mod(iter,batchsize) == 0;
    
    %if mod(iter,100) == 0; fprintf('%d..',iter); end
    
    [ftab(iter),grad] = fun(x);
    grad = grad(:);
    
    m = beta1 * m + (1-beta1) * grad;
    v = beta2 * v + (1-beta2) * grad.^2;
    mhat = m / (1-beta1^iter);
    vhat = v / (1-beta2^iter);
    
    stepsize = master_stepsize.min + ...
        (master_stepsize.max - master_stepsize.min)*exp(-iter/master_stepsize.decay);

    x = x - stepsize .* mhat ./(sqrt(vhat) + fudge_factor); % update
    x = min(max(x,LB(:)),UB(:));

    % xtab(:,idx) = x;    % Store X
    xtab(:,iter) = x;   % Store X
        
    if isMinibatchEnd && iter >= MinIter
        xxp = linspace(-(batchsize-1)/2,(batchsize-1)/2,batchsize);
        [p,S] = polyfit(xxp,ftab(iter-batchsize+1:iter),1);
        slope = p(1);
        Rinv = inv(S.R); A = (Rinv*Rinv')*S.normr^2/S.df;
        slope_err = sqrt(A(1,1) + TolFun^2);
        slope_err_max = sqrt(A(1,1) + TolFun_max^2);
        
        % Check random walk distance as termination condition
        %dx = sqrt(sum((mean(xtab(:,1:batchsize),2) - mean(xtab(:,batchsize+(1:batchsize)),2)).^2/batchsize,1));
        dx = sqrt(sum((mean(xtab(:,iter-batchsize+1:iter),2) - mean(xtab(:,(iter-batchsize+1:iter)-batchsize),2)).^2/batchsize,1));
        
        % Termination conditions
        if ( dx < TolX && abs(slope)<slope_err_max || abs(slope)<slope_err && dx < TolX_max )
            break;
        end
    end
    
end

% close all; figure(1); plot(1:iter,f);
% iter
%[dx,slope,slope_err]
% [x'; mean(xtab,2)']
% pause

% if mod(iter,batchsize*2) == 0
%     x = mean(xtab(:,batchsize+(1:batchsize)),2);
% else
%     x = mean(xtab(:,1:batchsize),2);
% end
x = mean(xtab(:,iter-batchsize+1:iter),2);
f = mean(ftab(iter-batchsize+1:iter));

xtab = xtab(:,1:iter);
ftab = ftab(1:iter);

x = reshape(x,size(x0));

if size(x,2) > 1; xtab = xtab'; end   % Transpose