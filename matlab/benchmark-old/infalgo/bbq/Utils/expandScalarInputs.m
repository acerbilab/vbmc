function [arg1, arg2, arg3, arg4, insize] = ...
    expandScalarInputs(arg1, arg2, arg3, arg4)

% If any of the inputs arg1, arg2, arg3, or arg4 are scalars, expand
% them to match the size of the others.  If the non-scalar inputs do not
% match in size, quit and return insize = []. Also check for consistency
% of NaN positions, and copy NaN into those same positions for arguments
% which are expanded from scalar values.

% Copyright 2005-2009 The MathWorks, Inc.
% $Revision: 1.1.6.3 $  $Date: 2009/07/06 20:36:15 $

% Check for matching argument sizes (scalars excepted)
nonscalar = ~[isscalar(arg1) isscalar(arg2) isscalar(arg3) isscalar(arg4)];
argsizes = {size(arg1) size(arg2) size(arg3) size(arg4)};
nonscalarsizes = argsizes(nonscalar);

if (numel(nonscalarsizes) >= 2) && ~isequal(nonscalarsizes{:})
    insize = [];
    return
end

% Assign input size (INSIZE) and expand scalar inputs if necessary
if ~any(nonscalar) || all(nonscalar)
    % Inputs all have the same size
    insize = argsizes{1};
else
    insize =  nonscalarsizes{1};
    args = {arg1, arg2, arg3, arg4};
    nanpos = cellfun(@isnan,args(nonscalar),'UniformOutput',false);
    nanpos = nanpos{1};

    if isscalar(arg1)
        arg1 = repmat(arg1, insize);
        arg1(nanpos) = NaN;
    end
    
    if isscalar(arg2)
        arg2 = repmat(arg2, insize);
        arg2(nanpos) = NaN;
    else
        assert(isequal(isnan(arg2),nanpos), ...
            'map:expandScalarInputs:mismatchedNanSeparators2', ...
            'Non-scalar inputs must have NaN-separators in identical positions.')       
    end
    
    if isscalar(arg3)
        arg3 = repmat(arg3, insize);
        arg3(nanpos) = NaN;
    else
        assert(isequal(isnan(arg3),nanpos), ...
            'map:expandScalarInputs:mismatchedNanSeparators3', ...
            'Non-scalar inputs must have NaN-separators in identical positions.')       
    end
    
    if isscalar(arg4)
        arg4 = repmat(arg4, insize);
        arg4(nanpos) = NaN;
    else
        assert(isequal(isnan(arg4),nanpos), ...
            'map:expandScalarInputs:mismatchedNanSeparators4', ...
            'Non-scalar inputs must have NaN-separators in identical positions.')       
    end
end
