dim = 2;
num = 100;

num_farpts = 10;

clf
tic;
% X = rand(num,dim);
% side = linspace(0,1,round(sqrt(num)))';
% X = allcombs([side, side]);
X = [1 1; 4 5; 6 8]/10;


hold on;
centres = far_pts(X, [zeros(1,dim); ones(1,dim)], num_farpts);

plot(centres(:,1),centres(:,2),'rx','MarkerSize',12,'LineWidth',2)
plot(X(:,1),X(:,2),'k.');
toc