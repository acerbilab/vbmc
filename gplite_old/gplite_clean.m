function gp = gplite_clean(gp)
%GPLITE_CLEAN Remove auxiliary info from lite GP struct (less memory usage).
%   GP = GPLITE_CLEAN(GP) removes auxiliary computational structs from
%   the GP. These can be reconstructed via a call to GPLITE_POST.
%
%   See also GPLITE_POST.

if ~isempty(gp) && isfield(gp,'post')
    post0 = struct('hyp',[]);
    
    for iG = 1:numel(gp)
        Ns = numel(gp(iG).post);
        postnew = post0;
        for iS = 1:Ns
            post_tmp = post0;
            for ff = fieldnames(post0)
                post_tmp.(ff{:}) = gp(iG).post(iS).(ff{:});
            end
            postnew(iS) = post_tmp;            
        end
        gp(iG).post = postnew;
    end
end