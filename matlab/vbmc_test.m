function y = vbmc_test(x,logflag)

if nargin < 2 || isempty(logflag); logflag = 0; end

D = numel(x);
sigma = 1;
df = 8;

if logflag
    mu = log(300)*ones(1,D);
    y = log(0.5*(mvtpdf((log(x)-mu)/sigma,eye(D),df) + ...
        mvtpdf((log(x)+mu)/sigma,eye(D),df))) - sum(log(x));
else    
    mu = ones(1,D);
    y = log(0.5*(mvtpdf((x-mu)/sigma,eye(D),df) + ...
        mvtpdf((x+mu)/sigma,eye(D),df)));
end

end