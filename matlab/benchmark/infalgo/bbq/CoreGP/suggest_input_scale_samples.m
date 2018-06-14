% function input_scale_samples = suggest_input_scale_samples (bounds, ...
%                                                     num_samples)
%
% calculate heuristic estimates for the input scale of a GP over
% the given box, assuming a grid of specified size in each dimension
%
% _arguments_
%       bounds: an 2xd set of lower and upper bounds for each dimension
%  num_samples: the number of samples to take in each dimension
%
% _returns_
%  input_scale_samples: an (num_samples)xd set of suggested input
%                       scale samples in each dimension
%
% author: roman garnett
%   date: 17 august 2008

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

function input_scale_samples = suggest_input_scale_samples (bounds, ...
	num_samples)
  
  dim = size(bounds, 2);

  % determine the volume of an n-ball in the given dimension
  sphere_volume_constant = pi^(dim / 2) / gamma(dim / 2 + 1);
  radius = (num_samples^dim * sphere_volume_constant) ^ (-1 / dim);
  
  % ensure that the calculated input scales guarantee that every
  % point will be somewhere between five and one input scales away
  % from a sample, assuming "good" samples
  unscaled_input_scale_samples = repmat(linspace(radius / 5, radius, ...
                                                 num_samples)', [1 dim]);
  
  % adjust samples to reflect actual ranges
  scaling_constants = repmat(diff(bounds), [num_samples 1]);
  input_scale_samples = unscaled_input_scale_samples .* scaling_constants;