function outwarp_test(outfun)
%OUTWARP_TEST Test correct implementation of an output warping function.

% Generate random observations
N = randi(50);
y = rand(N,1)*10;

[Noutwarp,info] = outfun('info',y);

% Generate random hyperparameters from plausible box
PLB = info.PLB(:);
PUB = info.PUB(:);
hyp = rand(Noutwarp,1).*(PUB - PLB) + PLB;

hyp

fprintf('---------------------------------------------------------------------------------\n');
fprintf('Check error on inverse of output warping function...\n\n');

sum(abs(y - outfun(hyp,outfun(hyp,y),'inv')))

fprintf('---------------------------------------------------------------------------------\n');
fprintf('Check 1st-order derivative of output warping function...\n\n');

yy = y(randi(N));
derivcheck(@(t) f(t,hyp,outfun),yy);

fprintf('---------------------------------------------------------------------------------\n');
fprintf('Check gradient of output warping function wrt hyperparameters...\n\n');

derivcheck(@(hyp_) f2(yy,hyp_,outfun),hyp);

fprintf('---------------------------------------------------------------------------------\n');
fprintf('Check gradient of derivative of output warping function wrt hyperparameters...\n\n');

derivcheck(@(hyp_) f3(yy,hyp_,outfun),hyp);



end

function [y,dy] = f(t,hyp,outfun)
    [y,dy] = outfun(hyp,t);
end

function [y,dy] = f2(y,hyp,outfun)
    [y,~,dy] = outfun(hyp,y);
end

function [y,dy] = f3(y,hyp,outfun)
    [~,y,~,dy] = outfun(hyp,y);
end