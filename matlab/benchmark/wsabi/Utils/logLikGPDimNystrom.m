function [nll] = logLikGPDimNystrom( xx, dat, hyp, numEigs )
outl = hyp(1);
inl  = hyp(2:end);
dim = size(xx,2);
dKxx = cell(1,length(hyp));
dll = zeros(size(hyp));
noise = 1e-7;


if any(isnan(hyp))
    keyboard;
end

numEigs = min(numEigs, size(xx,1));

idx = randperm( length(xx(:,1)) );

xxScaled = xx .* repmat(1./sqrt(exp(inl)),length(xx(:,1)),1);

xxuScaled = xxScaled( idx(1:numEigs), : );
xxsScaled = xxScaled;

A1 = pdist2_squared_fast(xxuScaled,xxuScaled);
A2 = pdist2_squared_fast(xxsScaled,xxuScaled);

Kuu = exp(2*outl) * 1/sqrt(det(2*pi*diag(exp(inl)))) * (exp( -0.5 * A1 ) + noise*eye(size(A1)));
Ksu = exp(2*outl) * 1/sqrt(det(2*pi*diag(exp(inl)))) * (exp( -0.5 * A2 ) + noise*eye(size(A2)));

[eVec, eVal] = eig(Kuu);
eVal2 = (eVal ./ eVal(1)) - 1/100;
[~,idx] = min(abs(eVal2));

eVal = eVal(1:idx,1:idx);
eVec = eVec(:,1:idx);

eVec = Ksu * (repmat(sqrt(numEigs) ./ diag(eVal)',length(eVal(:,1)),1) .* eVec); 
eVal = eVal / numEigs;

Z = noise * diag(1./diag(eVal)) + eVec'*eVec + noise*(eye(numEigs));

cholZ = chol(Z);

logKxx = (length( Ksu(:,1) ) - numEigs)*log(noise) + 2*sum(log(diag(cholZ))) + sum(log(diag(eVal)));
datKxxInvDat = (dat'*dat - dat'*eVec*(cholZ \ (cholZ' \ eVec'))*dat)*(1/noise);

%NeglogLik
nll = (1/2)*logKxx + (1/2)*datKxxInvDat + (length(xx(:,1))/2)*log(2*pi) + 0.5*sum(hyp.^2); 

if isnan(nll) || isinf(nll)
    keyboard;
end


% if nargout > 1
% 	dKxx{1} = 2*Kxx;
% 	for i = 2:length(hyp)
% 		dKxx{i} = -0.5*Kxx + 0.5 * Kxx .*  (pdist2_squared_fast(xx(:,i-1),xx(:,i-1)) / exp(inl(i-1)));
% 	end

% 	for i = 1:length(hyp)
% 		dll(i) = 0.5*trace((cholKxx \(cholKxx' \ dKxx{i}))) - 0.5*dat'*(cholKxx\(cholKxx' \ dKxx{i})) * (cholKxx\(cholKxx' \ dat)) + hyp(i);
% 	end
% end
end