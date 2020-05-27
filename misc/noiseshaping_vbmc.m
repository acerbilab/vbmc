function s2s = noiseshaping_vbmc(s2,y,options)
%NOISESHAPING_VBMC Increase noise for low-density points.

TolScale = 1e10;

if isempty(s2); s2 = options.TolGPNoise^2*ones(size(y)); end

deltay = max(0, max(y) - y - options.NoiseShapingThreshold); 
sn2extra = (options.NoiseShapingFactor*deltay).^2;

s2s = s2 + sn2extra;

maxs2 = min(s2s)*TolScale;
s2s = min(s2s,maxs2);
