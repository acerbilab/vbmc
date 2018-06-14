% function hyper_scale = get_hyper_scale(covvy, method)
%
% select the hyper scale to use at a particular time when
% performing BMC
%
% _arguments_
%
%   coovy: the covariance structure of interest
%  method: a string representing the method to use for selection:
%          * 'mle': peform maximum likelihood estimation
%
% _returns_
%
%  hyper_scale: a 1xd vector representing the chosen set of hyper^2
%               parameters
%
% author: roman garnett
%   date: 18 august 2008

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

function index = get_hyper_scale(covvy, method)

switch lower(method)
    case 'mle'
        [maximum, index] = max([covvy.hyper2samples.logL]);
    %	hyper_scale = covvy.hyper2samples(index).hyper2parameters;
    case 'tilda_mle'
        [maximum, index] = max([covvy.hyper2samples.tilda_logL]);
end
