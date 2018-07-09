%function [value gradient] = calculate_sample_uncertainty (means, SDs, ...
%  points) 
%
% calculates a value that can be used to minimize the uncertainty of the 
% BMC integral estimate in M. Osborne, "Gaussian Processes for Prediction,"
% 2007, equation 3.7.13
%
% _arguments_
%       means: a column vector representing the mean estimate for the points
%         SDs: a column vector representing the SDs of the estimates
%      points: a n x dim matrix containing the putative sample points
%
% _returns_
%    value: a value appropriate for minimizing the uncertainty
% gradient: an n x d matrix containing the partial derivatives of the form
%           [d/dx_1 point_1 d/dx2 point_1 ... d/dxd point_1; 
%            d/dx_1 point_2 ... d/dx_d point_n]
%
% author: michael osborne
%   date: 17 march 2009

% Copyright (c) 2009, Michael Osborne <mosb@robots.ox.ac.uk>
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


function [value gradient] = calculate_sample_uncertainty (means, SDs, points) 

gradient_required = (nargout > 1);

dimension = length(means);
num_samples = numel(points) / dimension;

nSDs = sqrt(2) * SDs;

points = reshape(points, [dimension num_samples]);

K_p_p = ones(num_samples);
n_m_p = ones(1, num_samples);
for d = 1:dimension
    points_d = points(d,:)';
    
    K_p_p = K_p_p .* ...
        matrify(@(a,b) normpdf(a, b, SDs(d)), points_d, points_d);
    n_m_p = n_m_p .* ...
        normpdf(points_d, means(d), nSDs(d))';
end

try
    chol_K_p_p = chol(K_p_p);
catch
    value = inf;
    gradient = points*inf;
    return
end
invKn = solve_chol(chol_K_p_p, n_m_p');
value = - n_m_p * invKn;

if (gradient_required)
     
    dn_m_p = (repmat(means, 1, num_samples) - points)...
                ./ repmat(nSDs.^2, 1, num_samples)...
                .* repmat(n_m_p, dimension, 1);
            
    first_term = -2 * dn_m_p .* repmat(invKn', dimension, 1);
    
    template = zeros(num_samples, num_samples, num_samples);
    for deriv_sample = 1:num_samples
        template(deriv_sample, :, deriv_sample) = 1;
        template(:, deriv_sample, deriv_sample) = -1;
    end
    template = reshape(template, num_samples, num_samples^2);
    
    dK_p_p = repmat(K_p_p, dimension, num_samples) ./ ...
      kron2d(SDs.^2, ones(num_samples, num_samples^2));
    for deriv_dim = 1:dimension
        inds = (deriv_dim - 1) * num_samples + (1:num_samples);
        dKmat = bsxfun(@minus,points(deriv_dim, :), points(deriv_dim, :)');
        dK_p_p(inds, :) = dK_p_p(inds, :) .* ...
          repmat(dKmat, 1, num_samples) .* template;
        % The point that you are taking the derivative wrt gets the
        % negative.
    end   
    
    dK_p_p_cell = ...
      mat2cell(dK_p_p, num_samples * ones(dimension, 1), num_samples * ones(num_samples, 1));
    second_term = -cellfun(@(dKmat) invKn' * dKmat * invKn, dK_p_p_cell);
    
    %gradient = reshape(first_term + second_term, [1 dimension * num_samples]);
    gradient = first_term + second_term;
end