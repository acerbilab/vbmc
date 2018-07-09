close all
scrsz = get(0,'ScreenSize');
width=14;
height=14;

mean = zeros(1,2);
cov = rand(2);
%cov = diag(rand(2,1));
cov = cov'*cov;

% cov = [0.2748    0.3
%     0.3    0.6163];

u(1) = 1.5;
u(2) = 1;

l(1) = -1;
l(2) = -2;

censor=@(xs) (xs(:,1)<u(1)).*(xs(:,2)<u(2)).*(xs(:,1)>l(1)).*(xs(:,2)>l(2));

f=@(xs) mvnpdf(xs,mean,cov).*censor(xs);

num = 100;
vec = linspace(-3,3,num)';

[X,Y] = meshgrid2d(vec,vec);

xs = [X(:),Y(:)];

fs = reshape(f(xs),num,num);

figure
set(gcf,'Position',[1 1 width/(2*37.7)*scrsz(3) height/30*scrsz(4)])
hold
contour(X,Y,fs)
[V,D] = eig(cov);
%vec1 = V(:,1)'*sqrt(D(1,1));
%vec2 = V(:,2)'*sqrt(D(2,2));
%sigma = repmat(mean,3,1) + [zeros(1,2);vec2;-vec2];
%sigma = [mean',repmat(mean',1,2)+chol(cov),repmat(mean',1,2)-chol(cov)]';
%plot(sigma(:,1),sigma(:,2),'k+','MarkerSize',10)

vec1 = V(:,1)';
vec2 = V(:,2)';

dim = 2;

means = repmat(mean,dim,1);

ud(1) = vec1*(mean+min((u(1)-mean(1))./vec1).*vec1)'

ud = sum(V.*(means+repmat(min(repmat(u-mean,dim,1)./V),dim,1).*V));
ld = sum(V.*(means+repmat(max(repmat(l-mean,dim,1)./V),dim,1).*V));