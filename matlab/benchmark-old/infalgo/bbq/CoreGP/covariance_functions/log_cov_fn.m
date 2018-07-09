function K = log_cov_fn(hyperparams)

T=exp(hyperparams(1));
L=exp(hyperparams(2));

Kfn= @(ax,bx) covfn('log',ax,bx,T,L);
K= @(Xs1,Xs2) matrify(Kfn,Xs1,Xs2);