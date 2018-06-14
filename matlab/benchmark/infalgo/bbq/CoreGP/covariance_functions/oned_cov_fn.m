function [K,DK,DDK] = oned_cov_fn(hyperparams)

T=exp(hyperparams(1));
L=exp(hyperparams(2));

Kfn= @(ax,bx) L^2*covfn('sqdexp',ax,bx,T,1);
DKfn= @(ax,bx) L^2*derivcovfn('sqdexp',ax,bx,T,1);
DDKfn= @(ax,bx) L^2*dblderivcovfn('sqdexp',ax,bx,T,1);

K= @(Xs1,Xs2) matrify(Kfn,Xs1,Xs2);
DK= @(Xs1,Xs2) {matrify(DKfn,Xs1,Xs2)};
DDK= @(Xs1,Xs2) {matrify(DDKfn,Xs1,Xs2)};