function c = colorbrew( i )
%
% Nice colors taken from 
% http://colorbrewer2.org/
%
% David Duvenaud
% March 2012

c_array(1, :) = [ 228, 26, 28 ];   % red
c_array(2, :) = [ 55, 126, 184 ];  % blue
c_array(3, :) = [ 77, 175, 74 ];   % green
c_array(5, :) = [ 152, 78, 163 ];  % purple
c_array(6, :) = [ 255, 127, 0 ];   % orange
c_array(7, :) = [ 255, 255, 51 ];  % yellow
c_array(8, :) = [ 166, 86, 40 ];   % brown
c_array(9, :) = [ 247, 129, 191 ]; % pink
c_array(10, :) = [ 153, 153, 153]; % grey

c = c_array( mod(i - 1, 10) + 1, : ) ./ 255;
end
