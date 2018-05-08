function enhance = enhanceline(last,options)
%BOLDLINE Enhance a subset of the plotted lines

enhance = 0;

if ~isfield(options,'EnhanceLine') || isempty(options.EnhanceLine)
    options.EnhanceLine = 0;
end

if options.EnhanceLine
    if ischar(options.EnhanceLine) && strcmpi(options.EnhanceLine,'last')
        enhance = last;
    elseif ischar(options.EnhanceLine) && strcmpi(options.EnhanceLine,'first')
        enhance = 1;
    elseif isnumeric(options.EnhanceLine)
        enhance = options.EnhanceLine;
    end
end

end