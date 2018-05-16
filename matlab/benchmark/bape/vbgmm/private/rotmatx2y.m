function R = rotmatx2y(x,y)
%ROTATEX2Y Return rotation matrix between two vectors.
%
%   R = ROTMATX2Y(X,Y) returns the rotation matrix that maps the direction
%   pointed by X to the direction pointed by Y (R maps X to Y if they are
%   unit vectors).
%

% Based on code by Stephen Montgomery-Smith.
% http://math.stackexchange.com/a/598782/325484

x = x(:);
y = y(:);
u = x/norm(x);
v = y-u'*y*u;
v = v/norm(v);
cost = x'*y / (norm(x)*norm(y));
sint = sqrt(1-cost^2);
R = eye(numel(x))-u*u'-v*v' + [u,v]* [cost,-sint; sint,cost]*[u,v]';

end