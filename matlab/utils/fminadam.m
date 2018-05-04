function [x,f] = fminadam(fun,x0,LB,UB,TolFun)

if nargin < 3; LB = []; end
if nargin < 4; UB = []; end
if nargin < 5 || isempty(TolFun); TolFun = 0.001; end

%% Adam with momentum
fudge_factor = sqrt(eps);

master_stepsize_max = 0.1;
master_stepsize_min = 0.001;
master_stepsize_decay = 200;
beta1 = 0.9;
beta2 = 0.999;
batchsize = 20;
TolX = 0.001;
TolX_max = 0.1;
TolFun_max = TolFun*100;

max_iter = 1e4;
min_iter = batchsize*2;

nvars = numel(x0);
if isempty(LB); LB = -Inf(nvars,1); end
if isempty(UB); UB = Inf(nvars,1); end

m = 0; v = 0;
xtab = zeros(nvars,batchsize*2);

x = x0(:);
f = NaN(1,max_iter);

for iter = 1:max_iter
    idx = mod(iter-1,batchsize*2) + 1;
    isMinibatchEnd = mod(iter,batchsize) == 0;
    
    %if mod(iter,100) == 0; fprintf('%d..',iter); end
    
    [f(iter),grad] = fun(x);
    grad = grad(:);
    
    m = beta1 * m + (1-beta1) * grad;
    v = beta2 * v + (1-beta2) * grad.^2;
    mhat = m / (1-beta1^iter);
    vhat = v / (1-beta2^iter);
    
    stepsize = master_stepsize_min + ...
        (master_stepsize_max - master_stepsize_min)*exp(-iter/master_stepsize_decay);

    x = x - stepsize * mhat ./(sqrt(vhat) + fudge_factor); % update
    x = min(max(x,LB(:)),UB(:));

    xtab(:,idx) = x;    % Store X
        
    if isMinibatchEnd && iter >= min_iter
        xxp = linspace(-(batchsize-1)/2,(batchsize-1)/2,batchsize);
        [p,S] = polyfit(xxp,f(iter-batchsize+1:iter),1);
        slope = p(1);
        Rinv = inv(S.R); A = (Rinv*Rinv')*S.normr^2/S.df;
        slope_err = sqrt(A(1,1) + TolFun^2);
        slope_err_max = sqrt(A(1,1) + TolFun_max^2);
        
        % Check random walk distance as termination condition
        dx = sqrt(sum((mean(xtab(:,1:batchsize),2) - mean(xtab(:,batchsize+(1:batchsize)),2)).^2/batchsize,1));
        
        % Termination conditions
        if ( dx < TolX && abs(slope)<slope_err_max || abs(slope)<slope_err && dx < TolX_max )
            break;
        end
    end
    
end

% close all; figure(1); plot(1:iter,f);
%iter
%[dx,slope,slope_err]
% [x'; mean(xtab,2)']
% pause

if mod(iter,batchsize*2) == 0
    x = mean(xtab(:,batchsize+(1:batchsize)),2);
else
    x = mean(xtab(:,1:batchsize),2);
end
f = mean(f(iter-batchsize+1:iter));

x = reshape(x,size(x0));

