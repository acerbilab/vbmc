function [LB,UB,PLB,PUB] = boundscheck(x0,LB,UB,PLB,PUB,prnt)
%BOUNDSCHECK Initial check of bounds.

[N0,nvars] = size(x0);

% Expand scalar inputs to full vectors
if isscalar(LB); LB = LB*ones(1,nvars); end
if isscalar(UB); UB = UB*ones(1,nvars); end
if isscalar(PLB); PLB = PLB*ones(1,nvars); end
if isscalar(PUB); PUB = PUB*ones(1,nvars); end

if isempty(PLB) || isempty(PUB)
    if N0 > 1
        if prnt > 0
            fprintf('PLB and/or PUB not specified. Estimating plausible bounds from starting set X0...\n');
        end
        width = max(x0) - min(x0);
        if isempty(PLB)
            PLB = min(x0) - width/N0;
            PLB = max(PLB,LB);
        end
        if isempty(PUB)
            PUB = max(x0) + width/N0;
            PUB = min(PUB,UB);
        end
        
        idx = any(PLB == PUB);
        if any(idx)
            PLB(idx) = LB(idx);
            PUB(idx) = UB(idx);
            warning('vbmc:pbInitFailed', ...
                'Some plausible bounds could not be determined from starting set. Using hard upper/lower bounds for those instead.')
        end
    else
        warning('vbmc:pbUnspecified', ...
            'Plausible lower/upper bounds PLB and/or PUB not specified and X0 is not a valid starting set. Using hard upper/lower bounds instead.');
        if isempty(PLB); PLB = LB; end
        if isempty(PUB); PUB = UB; end
    end
end

% Test that all vectors have the same length
if any([numel(LB),numel(UB),numel(PLB),numel(PUB)] ~= nvars)
    error('All input vectors (X0, LB, UB, PLB, PUB), if specified, need to have the same size.');
end

% Test that all vectors are row vectors
if ~isvector(LB) || ~isvector(UB) || ~isvector(PLB) || ~isvector(PUB) ...
        || size(LB,1) ~= 1 || size(UB,1) ~= 1 || size(PLB,1) ~= 1 || size(PUB,1) ~= 1
    error('All input vectors LB, UB, PLB, PUB, if specified, should be row vectors.');
end

% Test that plausible bounds are finite
if ~all(isfinite(([PLB, PUB]))) 
    error('Plausible interval bounds PLB and PUB need to be finite.');
end

% Test that all vectors are real-valued
if ~isreal([x0(:)', LB, UB, PLB, PUB])
    error('All input vectors should be real-valued.');
end

% Test that plausible bounds are different
if any(PLB == PUB)
    error('vbmc:MatchingPB', ...
        'For all variables, plausible lower and upper bounds need to be distinct.')
end

% Check that all X0 are inside the bounds
if any(any(bsxfun(@le,x0,LB))) || any(any(bsxfun(@ge,x0,UB)))
    error('vbmc:InitialPointsNotInsideBounds', ...
        'The starting points X0 are not strictly inside the provided hard bounds LB and UB.');
end

% Check that all X0 are inside the plausible bounds, move them otherwise
if any(any(bsxfun(@le,x0,PLB))) || any(any(bsxfun(@ge,x0,PUB)))
    warning('vbmc:InitialPointsOutsidePB', ...
        'The starting points X0 are not inside the provided plausible bounds PLB and PUB. Expanding the plausible bounds...');
    PLB = min(PLB,min(x0,[],1));
    PUB = max(PUB,max(x0,[],1));
end

% Test order of bounds
ordidx = LB <= PLB & PLB < PUB & PUB <= UB;
if any(~ordidx)
    error('vbmc:StrictBounds', ...
        'For each variable, hard and plausible bounds should respect the ordering LB <= PLB < PUB <= UB.');
end

% Fixed variables (all bounds equal)
fixidx = (LB == UB) & (UB == PLB) & (PLB == PUB);
if any(fixidx)
    error('vbmc:FixedVariables', ...
        'VBMC does not support fixed variables. Lower and upper bounds should be different.');
end