function [x0,LB,UB,PLB,PUB] = boundscheck(x0,LB,UB,PLB,PUB,prnt)
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

% Fixed variables (all bounds equal) are not supported
fixidx = (LB == UB) & (UB == PLB) & (PLB == PUB);
if any(fixidx)
    error('vbmc:FixedVariables', ...
        'VBMC does not support fixed variables. Lower and upper bounds should be different.');
end

% Test that plausible bounds are different
if any(PLB == PUB)
    error('vbmc:MatchingPB', ...
        'For all variables, plausible lower and upper bounds need to be distinct.')
end

% Check that all X0 are inside the bounds
if any(any(bsxfun(@lt,x0,LB))) || any(any(bsxfun(@gt,x0,UB)))
    error('vbmc:InitialPointsNotInsideBounds', ...
        'The starting points X0 are not inside the provided hard bounds LB and UB.');
end

% Compute "effective" bounds (slightly inside provided hard bounds)
bounds_range = UB - LB;
bounds_range(isinf(bounds_range)) = 1e3;
scale_factor = 1e-3;
LB_eff = LB + scale_factor*bounds_range;
LB_eff(abs(LB) <= realmin) = scale_factor*bounds_range(abs(LB) <= realmin);
UB_eff = UB - scale_factor*bounds_range;
UB_eff(abs(UB) <= realmin) = -scale_factor*bounds_range(abs(UB) <= realmin);
LB_eff(isinf(LB)) = LB(isinf(LB));  % Infinities stay the same
UB_eff(isinf(UB)) = UB(isinf(UB));

if any(LB_eff >= UB_eff)
    error('vbmc:StrictBoundsTooClose', ...
        'Hard bounds LB and UB are numerically too close. Make them more separate.');
end

% Fix when provided X0 are almost on the bounds -- move them inside
if any(any(bsxfun(@le,x0,LB_eff))) || any(any(bsxfun(@ge,x0,UB_eff)))
    warning('vbmc:InitialPointsTooClosePB', ...
        'The starting points X0 are on or numerically too close to the hard bounds LB and UB. Moving the initial points more inside...');
    x0 = bsxfun(@max, bsxfun(@min,x0,UB_eff), LB_eff);
end

% Test order of bounds (permissive)
ordidx = LB <= PLB & PLB < PUB & PUB <= UB;
if any(~ordidx)
    error('vbmc:StrictBounds', ...
        'For each variable, hard and plausible bounds should respect the ordering LB < PLB < PUB < UB.');
end

% Test that plausible bounds are reasonably separated from hard bounds
ordidx = LB_eff < PLB & PUB < UB_eff;
if any(~ordidx)
    warning('vbmc:TooCloseBounds', ...
        'For each variable, hard and plausible bounds should not be too close. Moving plausible bounds.');
    PLB = max(PLB,LB_eff);
    PUB = min(PUB,UB_eff);    
end

% Check that all X0 are inside the plausible bounds, move bounds otherwise
if any(any(bsxfun(@le,x0,PLB))) || any(any(bsxfun(@ge,x0,PUB)))
    warning('vbmc:InitialPointsOutsidePB', ...
        'The starting points X0 are not inside the provided plausible bounds PLB and PUB. Expanding the plausible bounds...');
    PLB = min(PLB,min(x0,[],1));
    PUB = max(PUB,max(x0,[],1));
end

% Test order of bounds
ordidx = LB < PLB & PLB < PUB & PUB < UB;
if any(~ordidx)
    error('vbmc:StrictBounds', ...
        'For each variable, hard and plausible bounds should respect the ordering LB < PLB < PUB < UB.');
end



% Test that variables are either bounded or unbounded (not half-bounded)
halfbnd = (isinf(LB) & isfinite(UB)) | (isfinite(LB) & isinf(UB));
if any(halfbnd)
    error('vbmc:HalfBounds', ...
        'Each variable needs to be unbounded or bounded. Variables bounded only below/above are not supported.');    
end