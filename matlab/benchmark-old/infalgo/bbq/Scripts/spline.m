
b = 10;
L = 6;
a = 10;

%K = @(x, y) bsxfun(@min,x,y')/L;
%K = @(x, y) a*(2*b - 2*b*abs(bsxfun(@minus,x,y'))/L);
K = @(x, y) a^2 + (b^2 - 2*b^2*abs(bsxfun(@minus,x,y'))/L);


N = 1000;
%x = rand(N, 1) * 3 * l;
x = linspace(-L,L,N)';
Kmat = K(x,x);

% P = inv(Kmat);
% P(1:10, 1:10)

% y = mvnrnd(zeros(N, 1), Kmat);
% 
% plot(x, y, '.')

D = 5;
sigma = b * 1e-8;

x_d = sort((rand(D,1)-0.5) * 2*L/3);
y_d = b*sin(rand*x_d + 2*pi*rand);%a * (rand(D,1) - 0.5);

K_d = K(x_d, x_d);

vec = K(x, x_d)/(K_d + eye(D) * sigma^2);

m = vec * y_d;
C = Kmat - vec * K(x_d, x) + sigma^2 * eye(N);
C = 0.5*(C'+C);
V = diag(C);

figure(1)

y = mvnrnd(m, C);
clf
gp_plot(x, m, sqrt(V), x_d, y_d, x, y)

