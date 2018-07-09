function [s,opt,opt2] = sepn2(ls) 
if length(ls)<=1
    s=1;
else
    ls=sort(ls);
    s=min(ls(2:end)-ls(1:end-1));
end
switch nargout
    case {1,2}
        opt = 2;
    case 3
        opt = 4;
        opt2 = pi;
end