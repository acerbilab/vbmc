function [ mat ] = rot_matrix( theta )
%UNTITLED returns 2D rotation matrix defined by angle theta

mat = [cos(theta) -sin(theta); sin(theta) cos(theta)];

end

