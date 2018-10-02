function warpvars_test(nvars)

if nargin < 1 || isempty(nvars); nvars = 1; end


for iType = [0,3,9,10]
    
    fprintf('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n');
    fprintf('Testing transformation type %d...\n',iType);
    
    nvars = 1;
    switch iType
        case {0,10}
            LB = -Inf; UB = Inf; PLB = -10; PUB = 10;
        case {3,9}
            LB = -9; UB = 4; PLB = -8.99; PUB = 3.99;
    end

    trinfo = warpvars(nvars,LB,UB);
    trinfo.type = iType*ones(1,nvars);
    trinfo.alpha = exp(2*rand(1,nvars));
    trinfo.beta = exp(2*rand(1,nvars));
    trinfo.mu = 0.5*(PUB+PLB);
    trinfo.delta = (PUB-PLB);

    x = linspace(PLB,PUB,101);
    x2 = warpvars(warpvars(x,'dir',trinfo),'inv',trinfo);    
    
    fprintf('Maximum error for identity transform f^-1(f(x)): %.g.\n\n',max(abs(x - x2)));    

    fprintf('Checking derivative and log derivative:\n\n');
    x0 = rand(1,nvars).*(PUB-PLB)+PLB;
    derivcheck(@(x) fun(x,trinfo,0),x0,1);
    derivcheck(@(x) fun(x,trinfo,1),x0,1);
    
    if any(iType == [9 10])
        fprintf('Checking derivatives wrt warping parameters:\n\n');
        theta0 = 3*randn(1,2);
        derivcheck(@(theta) funfirst(theta,x0,trinfo),theta0',0);
        derivcheck(@(theta) funmixed(theta,x0,trinfo),theta0',0);
    end
    
end
    
    
% if nvars == 1
%     x = linspace(LB+sqrt(eps),UB-sqrt(eps),101);
%     x2 = warpvars(warpvars(x,'dir',trinfo),'inv',trinfo);
%     max(abs(x - x2))
% 
%     x0 = rand(1,nvars).*(UB-LB)+LB;
%     derivcheck(@(x) fun(x,trinfo),x0,1);
% else
%     N = 10;
%     [Q,R] = qr(randn(nvars));
%     if det(Q) < 0; Q(:,1) = -Q(:,1); end
%     trinfo.R_mat = Q;
%     % trinfo.R_mat = eye(Nvars);
%     % trinfo.scale = exp(randn(1,Nvars));
%     
%     x = randn(N,nvars);    
%     x2 = warpvars(warpvars(x,'dir',trinfo),'inv',trinfo);
% 
%     x - x2
%     
%     
%     
%     x0 = 0.1*rand(1,nvars).*(UB-LB)+LB;    
%     x0t = warpvars(x0,'dir',trinfo);
%     
%     derivcheck(@(x) fun(x,trinfo),x0,1);
%     derivcheck(@(x) invfun(x,trinfo),x0t,1);
%     
% end


% x0 = randn(1,Nvars);
% derivcheck(@(theta) funfirst(theta,x0,trinfo),0.1*randn(1,2),0);


end

function [y,dy] = fun(x,trinfo,logflag)

if nargin < 3 || isempty(logflag); logflag = 0; end

y = warpvars(x,'dir',trinfo);
% dy = warpvars(y,'g',trinfo);

if logflag
    dy = exp(-warpvars(y,'logpdf',trinfo));
else
    dy = 1./warpvars(y,'pdf',trinfo);
end

end

function [y,dy] = invfun(x,trinfo)

y = warpvars(x,'inv',trinfo);
dy = warpvars(x,'r',trinfo);
% dy = exp(-warpvars(y,'logpdf',trinfo));

end


function [y,dy] = funfirst(theta,x,trinfo)

nvars = numel(trinfo.lb_orig);
theta = exp(theta);

trinfo.alpha(1) = theta(1);
trinfo.beta(1) = theta(2);

y = warpvars(x,'d',trinfo);
dy = warpvars(y,'f',trinfo); 

dy = dy([1;nvars+1]) .* theta(:)';
% dy = exp(-warpvars(y,'logpdf',trinfo));

end


function [dy,ddy] = funmixed(theta,x,trinfo)

nvars = numel(trinfo.lb_orig);
theta = exp(theta);

trinfo.alpha(1) = theta(1);
trinfo.beta(1) = theta(2);

y = warpvars(x,'d',trinfo);
dy = exp(-warpvars(y,'logpdf',trinfo));

ddy = warpvars(y,'m',trinfo);
ddy = ddy([1;nvars+1]) .* theta(:)';

end
