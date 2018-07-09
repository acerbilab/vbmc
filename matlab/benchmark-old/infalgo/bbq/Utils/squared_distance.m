% function distances = squared_distance(A)
%          distances = squared_distance(A, B)
%
% finds squared euclidean distance or squared Mahalanobis distance between 
% two sets of points
%
% _arguments_
%     A: an nxd set of points of interest
%     B: an mxd set of points of interest
% Sigma: if present, calculates the squared Mahalanobis distance using
%        Sigma, a dxd matrix
%
% _returns_
% distances: the nxm matrix of squared distances between the points in
%            A and B

% author: roman garnett
%   date: 28 june 2008
%
% Copyright (c) 2008, Roman Garnett <rgarnett@robots.ox.ac.uk>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.
% 
% THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
% WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
% ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

function distances = squared_distance(A, B, Sigma)

if (nargin == 2)
  A_length = sum(A .* A, 2);
  B_length = sum(B .* B, 2);
  distances = repmat(A_length, 1, numel(B_length)) + ...
              repmat(B_length', numel(A_length), 1) - ...
              2 * A * B';
  distances(distances<0)=0;
  return
end

if size(Sigma,1)==1
    
    ASigma_length = A.^2 * Sigma'.^-2;
    BSigma_length = B.^2 * Sigma'.^-2;
    distances = repmat(ASigma_length, 1, numel(BSigma_length)) + ...
                repmat(BSigma_length', numel(ASigma_length), 1) - ...
                2 * A * diag(Sigma.^-2) * B';
    distances(distances<0)=0;
end


