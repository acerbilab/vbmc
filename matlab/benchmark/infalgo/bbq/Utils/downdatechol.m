function R=downdatechol(S,two)
% I don't want to have to do checks on this, but we have to have
% two as a contiguous block. Also obviously have to have S upper triangular.

one=1:(min(two)-1);
three=(max(two)+1):length(S);
twothree=min(two):length(S);
newthree=three-length(two);

R = zeros(length(S)-length(two));

R(one,one)=S(one,one);
R(one,newthree)=S(one,three);
R33=S(twothree,twothree);
for drop=two
    % unfortunately, cholupdate can only handle column
    % corrn's, which makes my life just that much more difficult
    corrn=R33(1,2:end)';
    R33=cholupdate(R33(2:end,2:end),corrn); % NB: cholupdate is a proprietary matlab function
end

R(newthree,newthree)=R33;
