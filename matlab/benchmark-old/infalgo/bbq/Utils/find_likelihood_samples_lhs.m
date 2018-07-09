function [samples best_uncertainty] = ...
  find_likelihood_samples_lhs (means, variances, num_samples, ...
  allowed_time, minimum_condition)

dimension = size(means,1);
if (dimension ~= size(variances,1))
  error('length(variances) should equal length(means)!');
end

means = means(:);
sds = sqrt(variances(:));

samples = zeros(dimension, num_samples);

best_uncertainty = Inf;
start_time = cputime;

objective = @(x) calculate_sample_uncertainty(means, sds, x');

while (cputime < start_time + allowed_time)
  points = zeros(num_samples, dimension);
  
  nonzero_points = get_sample(num_samples - 1, dimension);
  
  offsets = normcdf(zeros(dimension, 1), means, sds)';

  points(2:end, :) = norminv(repmat(offsets, [(num_samples - 1) 1]) + ...
    (nonzero_points .* ...
    repmat(max(nonzero_points) - offsets, [(num_samples - 1) 1]))) .* ...
    repmat(sds', [(num_samples - 1) 1]) + ...
    repmat(means', [(num_samples - 1) 1]);

  K_p_p = ones(num_samples);
  for d = 1:dimension
    K_p_p = K_p_p .* ...
      matrify(@(a, b) normpdf(a, b, sds(d)), points(:, d), points(:, d));
  end
    if (cond(K_p_p) > minimum_condition)
    continue
  end

  uncertainty = objective(points);
  
  if (uncertainty < best_uncertainty)
    samples = points;
    best_uncertainty = uncertainty;
  end
end

if (best_uncertainty == Inf)
  error('could not find points with proper conditioning!');
end
 
function sample = get_sample (num_samples, dimension)

sample = rand(num_samples, dimension);
for i = 1:dimension
   sample(:, i) = rank(sample(:, i));
end
sample = sample - rand(size(sample));
sample = sample / num_samples;
   
% -----------------------
function r = rank(x)

% Similar to tiedrank, but no adjustment for ties here
[sx, rowidx] = sort(x);
r(rowidx) = 1:length(x);
r = r(:);
