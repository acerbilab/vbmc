% Ryan Turner rt324@cam.ac.uk

function idx = argmin(X, dim)

if nargin == 2
  [temp, idx] = min(X, dim);
else
  [temp, idx] = min(X);
end
