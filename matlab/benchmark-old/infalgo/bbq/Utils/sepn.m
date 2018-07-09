function s = sepn(ls) 
if length(ls)<=1
    s=1;
else
    s=(max(ls)-min(ls))/(length(ls)-1);
end