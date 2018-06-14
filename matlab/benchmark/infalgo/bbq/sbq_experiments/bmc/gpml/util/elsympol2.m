% Evaluate the order R elementary symmetric polynomial Newton's identity aka
% the Newton–Girard formulae: http://en.wikipedia.org/wiki/Newton's_identities
%
% Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-01-10.
%
% Modified by David Duvenaud to be more numerically stable, 2011-04-28.

function E = elsympol2(Z,R)
% evaluate 'power sums' of the individual terms in Z
sz = size(Z);
P = zeros([sz(1:2),R]);
Z_power = Z;
for r=1:R
    P(:,:,r) = sum(Z_power,3);
    Z_power = Z_power .* Z;    % Evaluate the power incrementally.
end

E = zeros([sz(1:2),R+1]);                   % E(:,:,r+1) yields polynomial r
E(:,:,1) = ones(sz(1:2)); if R==0, return, end  % init recursion
E(:,:,2) = P(:,:,1);      if R==1, return, end  % init recursion
for r=2:R
  for i=1:r
    E(:,:,r+1) = E(:,:,r+1) + (P(:,:,i).*E(:,:,r+1-i))*((-1)^(i-1)/r);
  end
end
