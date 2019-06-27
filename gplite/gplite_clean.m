function gp = gplite_clean(gp)
%GPLITE_CLEAN Remove auxiliary info from lite GP struct (less memory usage).
%   GP = GPLITE_CLEAN(GP) removes auxiliary computational structs from
%   the GP. These can be reconstructed via a call to GPLITE_POST.
%
%   See also GPLITE_POST.

if ~isempty(gp) && isfield(gp,'post')
    copyfields = {'hyp'};
    emptyfields = {'alpha','sW','L','sn2_mult','Lchol'};
    for ff = copyfields; post0.(ff{:}) = []; end
    for ff = emptyfields; post0.(ff{:}) = []; end
    
    for iG = 1:numel(gp)
        Ns = numel(gp(iG).post);
        postnew = post0;
        for iS = 1:Ns
            post_tmp = post0;
            for ff = copyfields
                post_tmp.(ff{:}) = gp(iG).post(iS).(ff{:});
            end
            postnew(iS) = post_tmp;            
        end
        gp(iG).post = postnew;
    end
end