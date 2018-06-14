function centres = metrickcentres(points, k)
% points is an n by d matrix
% k is the number of centres to be returned
% centres is a k by d matrix

[num_points dimension] = size(points);
centres = zeros(k, dimension);

in_set = false(num_points, 1);

% first choice is arbitrary
in_set(1) = true;

for i = 2:k
  distances_to_current_set = ...
    sq_dist(points(~in_set, :)', points(in_set, :)');
  [dummy, farthest] = max(min(distances_to_current_set, [], 2));
  out = find(~in_set);
  in_set(out(farthest)) = true;
end

centres = points(in_set, :);