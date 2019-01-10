function tf = vbmc_isavp(vp)
%VBMC_ISAVP True for VBMC variational posterior structures.
%   VBMC_ISAVP returns true if VP is a variational posterior structure
%   returned by VBMC and false otherwise.
%
%   See also VBMC.

if isstruct(vp)
    
    tf = true;    
    
    % Required fields for variational posterior
    vpfields = {'D','K','w','mu','sigma','lambda','trinfo', ...
        'optimize_mu','optimize_lambda','optimize_weights','bounds'};
    
    % Check that VP has all the required fields, otherwise quit
    ff = fields(vp);        
    for iField = 1:numel(vpfields)
        if ~any(strcmp(vpfields{iField},ff))
            tf = false;
            break;
        end
    end
       
else
    tf = false;
end