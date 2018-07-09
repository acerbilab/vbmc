function [samples,best_uncertainty] = ...
  find_likelihood_samples (means, variances, num_samples, allowed_time, try_hard)

if (nargin < 5)
  try_hard = false;
end

dimension = length(means);
if (dimension ~= length(variances))
  error('length(variances) should equal length(means)!');
end

sds = sqrt(variances);
samples = zeros(dimension, num_samples);
total_points_without_zero = dimension * (num_samples-1);

lower_bound = -3 * repmat(sds(:)', [1 num_samples-1]);
%lower_bound = zeros(1,total_points_without_zero);
upper_bound = 3 * repmat(sds(:)', [1 num_samples-1]);

best_uncertainty = Inf;
best_point = zeros(1, length(samples));
start_time = cputime;

objective=@(x) calculate_augmented_sample_uncertainty(x,means,sds);

while (cputime < start_time + allowed_time)
  

  
  starting_point_poorly_conditioned = true;
  while starting_point_poorly_conditioned
        % assume independence
        
%        lhsdesign(6,total_points_without_zero,'criterion','maximin','iterations',1000)
        
         starting_point = ...%abs
         ((randn(1, total_points_without_zero) + ...
             repmat(means(:)', [1 num_samples-1])) .* ...
             repmat(sds(:)', [1 num_samples-1]));

        points = reshape(starting_point, [dimension num_samples-1]);
        points = [zeros(dimension,1), points];
        K_p_p = ones(num_samples);
        for d=1:dimension
            points_d=points(d,:)';
            K_p_p = K_p_p.*...
                matrify(@(a,b) normpdf(a,b,sds(d)), points_d, points_d);
        end
        starting_point_poorly_conditioned = cond(K_p_p)>10^3;
  end

  options = optimset('Display','off');

  if ~(try_hard)
   options=optimset(options,'Algorithm','active-set'); 
  else
      % This seems to stop prematurely for some reason I cannot fathom.
    %options=optimset('Algorithm', 'trust-region-reflective','LargeScale','on','GradObj','on'); 
    options=optimset(options,'GradObj','on'); 
  end
  
  [solution uncertainty] = ...
    fmincon(objective, starting_point, [], [], [], [], lower_bound, upper_bound, [], options);
    solution=[zeros(1,dimension), solution];

  if (uncertainty < best_uncertainty)
    best_uncertainty = uncertainty;
    best_point = solution;
  end

    samples = reshape(best_point, [dimension num_samples]);


            
%         points = samples;
%         K_p_p = ones(num_samples);
%         for d=1:dimension
%             points_d=points(d,:)';
%             K_p_p = K_p_p.*...
%                 matrify(@(a,b) normpdf(a,b,sds(d)), points_d, points_d);
%         end
%         disp('after_optim')
%         cond(K_p_p)
end

function [unc,grad]=calculate_augmented_sample_uncertainty(x,means,sds)
dimension=length(means);
if nargout>1
    [unc,grad]=calculate_sample_uncertainty(means(:), sds(:), [zeros(1,dimension), x]);
    grad=grad(dimension+1:end);
else
    [unc]=calculate_sample_uncertainty(means(:), sds(:), [zeros(1,dimension), x]);
end

