function logp = logmvnpdf(x,mu,Sigma)
% Log of multivariate normal pdf.
%
% David Duvenaud
% January 2012.

dim = length(mu);
logdetcov = logdet(Sigma);
a = bsxfun(@minus, x, mu);
logp = (-dim/2)*log(2*pi) + (-.5)*logdetcov +...
    (-.5.*sum(bsxfun( @times, a / Sigma, a), 2));   % Evaluate for multiple inputs.
end

function ld = logdet(K)
    % returns the log-determinant of posdef matrix K.
    
    % This is probably horribly slow.
    ld = NaN;
    try
        ld = 2*sum(log(diag(chol(K))));
    catch e
        e;
    end
end
