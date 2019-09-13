function x = real2int_vbmc(x,trinfo,integervars)
%REAL2INT_VBMC Convert to integer-valued representation.

if ~any(integervars); return; end

xtemp = warpvars_vbmc(x,'inv',trinfo);
xtemp(:,integervars) = round(xtemp(:,integervars));
xtemp = warpvars_vbmc(xtemp,'d',trinfo);

x(:,integervars) = xtemp(:,integervars);

end