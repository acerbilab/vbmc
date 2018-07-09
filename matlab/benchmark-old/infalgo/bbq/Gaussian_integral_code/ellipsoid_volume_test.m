% generates num_points uniformly in [0, side_length]^dimension,
% applies squared exponential kernel with parameters
% [log_input_scale, log_output_scale], finds an axis-aligned
% bounding ellipsoid, and compares the volume of each.

% parameters
dimension        = 3;
num_points       = 100;
side_length      = 10;
log_input_scale  = log(1);
log_output_scale = log(1);
jitter           = 1e-10;

% formulas to calculate log ellipsoid volume
logdet = @(K) 2 * sum(log(diag(chol(K))));
log_volume = @(K) log(pi) + log(4) - log(3) + logdet(K) / 2;

% generate points
x = side_length * rand(num_points, dimension);

% calculate prior covariance
K_original = covSEiso([log_input_scale; log_output_scale], x);

% symmetrize and jitterize
K_original = (K_original + K_original') / 2;
K_original = K_original + jitter * eye(num_points);

original_log_volume = log_volume(K_original);

fprintf('log volume of original K: %0.3f\n', ...
        original_log_volume);

% solve diagonal sdp
fprintf('solving sdp...');
cvx_begin sdp quiet
  cvx_precision best
  variable K_diagonal(num_points, num_points) diagonal
  minimize(trace(K_diagonal))
  K_diagonal >= K_original
cvx_end
fprintf('done.\n');

diagonal_log_volume = log_volume(K_diagonal);

fprintf(['log volume of digonal K: %0.3f\n' ...
         'the axis-aligned ellipsoid is %0.3e times bigger.\n'], ...
        diagonal_log_volume, ...
        exp(diagonal_log_volume - original_log_volume));
