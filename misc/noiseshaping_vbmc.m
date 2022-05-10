function s2s = noiseshaping_vbmc(s2,y,options)
%NOISESHAPING_VBMC Increase noise for low-density points.

if isempty(s2); s2 = options.TolGPNoise^2*ones(size(y)); end

min_lnsigma = log(options.NoiseShapingMin);
med_lnsigma = log(options.NoiseShapingMed);
frac = min(1, (max(y) - y) / options.NoiseShapingThreshold);
sn2extra = exp(2*(min_lnsigma*(1 - frac) + frac*med_lnsigma));

deltay = max(0, max(y) - y - options.NoiseShapingThreshold); 
sn2extra = sn2extra + (options.NoiseShapingFactor*deltay).^2;

s2s = s2 + sn2extra;

% Excessive difference between low and high noise might cause numerical
% instabilities, so we give the option of capping the ratio
maxs2 = min(s2s)*options.NoiseShapingMaxRatio^2;
s2s = min(s2s,maxs2);