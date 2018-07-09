function y = har3(x)
% function y = har3(x)
% 
% 3d Hartmann function 
% Matlab Code by A. Hedar (Sep. 29, 2005).
% The number of variables n = 3.
% 
% constraints:
% 0 <= xi <= 1, i = 1,2,3
% global optimum at (0.114614, 0.555649, 0.852547), where har3 = 3.86278

a(:,2)=10.0*ones(4,1);
for j=1:2;
   a(2*j-1,1)=3.0; a(2*j,1)=0.1; 
   a(2*j-1,3)=30.0; a(2*j,3)=35.0; 
end
c(1)=1.0;c(2)=1.2;c(3)=3.0;c(4)=3.2;
p(1,1)=0.36890;p(1,2)=0.11700;p(1,3)=0.26730;
p(2,1)=0.46990;p(2,2)=0.43870;p(2,3)=0.74700;
p(3,1)=0.10910;p(3,2)=0.87320;p(3,3)=0.55470;
p(4,1)=0.03815;p(4,2)=0.57430;p(4,3)=0.88280;
s = 0;
for i=1:4;
   sm=0;
   for j=1:3;
      sm=sm+a(i,j)*(x(j)-p(i,j))^2;
   end
   s=s+c(i)*exp(-sm);
end
y = -s;
