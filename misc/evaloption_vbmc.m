function val = evaloption_vbmc(option,N)
%GETVALUE_VBMC Return option value that could be a function handle.

if isa(option,'function_handle')
    val = option(N);
else
    val = option;
end

end