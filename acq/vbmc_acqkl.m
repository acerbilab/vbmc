function acq = vbmc_acqkl(G,varG,G_old,varG_old)
%VBMC_ACQKL Maximum expected Kullback-Leibler divergence acquisition function.

%acq = -(0.5*(vardiagGs/vardiagG + vardiagG./vardiagGs) ...
%    + 0.5*(G-Gs).^2/vardiagG + 0.5*(G-Gs).^2./vardiagGs - 1);

acq = -0.5*(varG/varG_old ...
    + (G_old-G).^2/varG_old - 1 - log(varG/varG_old));

end