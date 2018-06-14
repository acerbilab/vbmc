function inds = find_list(x,y)
% find inds such that x(inds) is as close as possible to y

n = length(y);

inds = zeros(n,1);
for i = 1:n
   [~,inds(i)]=min(abs(x-y(i)));
end