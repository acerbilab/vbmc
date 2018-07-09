% function candidates = gradient_ascent(points, gradients, ...
%                                       length_scales, step_size)
%
% shift given points by one step of gradient ascent, optinally
% adjusting by given length scales
%
% _arguments_
%        points: an nxd set of d-dimensional points
%     gradients: an nxd set of gradients at the points specified in points
% length_scales: (optional) an 1xd vector of length scales for each dimension
%     step_size: (optional) the step size to use; the algorithm
%                will shift the points by step_size length scales in
%                the direction specified by the gradients
%
% _returns_
% candidates: the points in points shifted by one step of gradient ascent
%
% author: roman garnett
%   date: 16 august 2008

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

function [candidates new_estimates] = gradient_ascent(points, ...
                                    old_values, gradients, ...
                                    step_size, length_scales)

  [num_points dim] = size(points);

  % defaults
  if (nargin < 4); step_size = 1; end
  if (nargin < 5); length_scales = ones(1, dim); end

  scale_matrix = repmat(length_scales, [num_points 1]);
  
  zero_gradients=gradients==0;
  
  % scale gradients by length scales
  scaled_gradients = gradients .* scale_matrix;
  
  % Below we scale again, eliminating the units of the output (likelihood,
  % for example). scaled_gradients is then a direction, as desired.
  constants=repmat(max(abs(scaled_gradients),[],2),[1 dim]);
  
  scaled_gradients = scaled_gradients ./ constants;
  norms = repmat(sqrt(sum(scaled_gradients.^2, 2)), [1 dim]);
   direction = (scaled_gradients ./ norms);

  % shift by a portion of "an input scale"
  candidates = points + (step_size * scale_matrix) .* direction;
  candidates(zero_gradients)=points(zero_gradients);
	
% estimate new function values
if nargout>1
    new_estimates = old_values + sum(gradients .* (candidates - points),2);
end