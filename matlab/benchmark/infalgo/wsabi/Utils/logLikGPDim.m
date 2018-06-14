function [nll, dll] = logLikGPDim( xx, dat, hyp )
outl = hyp(1);
inl  = hyp(2:end);

dKxx = cell(1,length(hyp));
dll = zeros(size(hyp));

xxScaled = xx .* repmat(1./sqrt(exp(inl)),length(xx(:,1)),1);
A = pdist2_squared_fast(xxScaled,xxScaled);


Kxx = exp(2*outl) * 1/sqrt(det(2*pi*diag(exp(inl)))) * (exp( -0.5 * A ) + 1e-5*eye(size(A)));
Kxx = Kxx/2 + Kxx'/2;


cholKxx = jitter_chol(Kxx);


%logLik
nll = (1/2)*2*sum(log(diag(cholKxx))) + 0.5*dat'*(cholKxx\(cholKxx' \ dat)) + (length(xx(:,1))/2)*log(2*pi)+ 0.5*sum(hyp.^2);

if any(isnan(nll))
    keyboard;
end


if nargout > 1
dKxx{1} = 2*Kxx;
for i = 2:length(hyp)
    dKxx{i} = -0.5*Kxx + 0.5 * Kxx .*  (pdist2_squared_fast(xx(:,i-1),xx(:,i-1)) / exp(inl(i-1)));
end

for i = 1:length(hyp)
    dll(i) = 0.5*trace((cholKxx \(cholKxx' \ dKxx{i}))) - 0.5*dat'*(cholKxx\(cholKxx' \ dKxx{i})) * (cholKxx\(cholKxx' \ dat)) + hyp(i);
end
end
end