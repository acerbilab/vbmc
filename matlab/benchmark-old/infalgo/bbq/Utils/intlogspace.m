function vec = intlogspace(lo, hi, num) 
% generate a vector of num monotonically increasing integers, beginning at
% lo and ending with hi, that are logarithmically space (more towards lo).

if hi-lo <= num-1
    both = [hi,lo];
    vec = min(both):max(both);
    return;
elseif num==0
    vec=[];
end

vec = nan(1, num);
i = 1;
x0 = lo;
vec(i) = x0;
x1 = x0 * exp((log(hi) - log(x0))/(num-1));

while x1 - x0 < 1 && i <= num
    x0 = x0 + 1;
    i = i+1;

    vec(i) = x0;

    x1 = x0 * exp((log(hi) - log(x0))/(num-i));
end
vec(i+1:end) = round(exp(linspace(log(x1),log(hi), num-i)));