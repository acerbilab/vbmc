function candidates = ...
    find_candidates(points, requested, length_scales, num_scales)
% function candidates = ...
%     find_candidates(points, requested, length_scales)
% return random candidates removed by a num_scales input scales (in Euclidean
% distance) from points.

if nargin<4
    num_scales = 1;
end

[num_points, num_dims] = size(points);
directions = rand(requested, num_dims)-0.5;
directions_length = sqrt(sum(directions.^2, 2));
directions = bsxfun(@rdivide, directions, directions_length);
% scale by length_scales
directions = bsxfun(@times, directions, num_scales * length_scales);

start_inds = ceil(num_points * rand(requested, 1));
starts = points(start_inds,:);

candidates = starts + directions;