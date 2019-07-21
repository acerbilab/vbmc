function vp_real = vptrain2real(vp,entflag,options)
%VPTRAIN2REAL Convert training variational posterior to real one.

if nargin < 2 || isempty(entflag); entflag = false; end
if nargin < 3; options = []; end

if isfield(vp,'temperature') && ~isempty(vp.temperature)
    T = vp.temperature;
else
    T = 1;
end

if any(T == [2,3,4,5])
    PowerThreshold = 1e-5;
    [vp_real,lnZ_pow] = vbmc_power(vp,T,PowerThreshold);
    if isfield(vp_real,'stats') && ~isempty(vp_real.stats)    
        vp_real.stats.elbo = T*vp.stats.elbo + lnZ_pow;
        vp_real.stats.elbo_sd = T*vp.stats.elbo_sd;
        vp_real.stats.elogjoint_sd = T*vp.stats.elogjoint_sd;
        
        if entflag
            % Use deterministic approximation of the entropy
            H = entlb_vbmc(vp_real,0,1);
            varH = 0;            
            vp_real.stats.elogjoint = vp_real.stats.elbo - H;
            vp_real.stats.entropy = H;
            vp_real.stats.entropy_sd = sqrt(varH);
        else
            vp_real.stats.elogjoint = NaN;
            vp_real.stats.entropy = NaN;
            vp_real.stats.entropy_sd = NaN;            
        end
    end
else
    vp_real = vp;
end

