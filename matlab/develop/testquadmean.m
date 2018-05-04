function hyp = testquadmean

meanfun = 'negquad';

D = 1;
m0 = 10;
xm = 2*ones(1,D);
omega = sqrt(2)*ones(1,D);

X = linspace(-1,1,4)';
y = m0 - 0.5*sum((X-xm)./omega,2).^2 + randn(size(X,1),1);

gp = gplite_train(zeros(3*D+3),10,X,y,meanfun);

Xstar = linspace(-20,20,1001)';
[ym,ys2] = gplite_pred(gp,Xstar);

plot(Xstar,ym,'k-'); hold on;
plot(Xstar,ym+sqrt(ys2),'k:');
plot(Xstar,ym-sqrt(ys2),'k:');
scatter(X,y,'r.');

end