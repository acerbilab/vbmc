function [out, sqd_jitters] = improve_covariance_conditioning(in,importance,allowed_error,flag)
% function [out] = improve_covariance_conditioning(in,importance,allowed_error)
% in: potentially poorly conditioned covariance matrix
% importance: the relative importances of the variables associated with the
% rows/columns of in
% allowed_error: the allowed error in the output
% flag: if flag=='identify_problems', instead out is a logical vector
% specifying with points are problematic
% out: covariance matrix with improved conditioning
% sqd_jitters: the elements that were added to the diagonal of out
% A heuristic method of managing potential issues with the conditioning of
% covariance matrix. If in contains nan elements, it is assumed that they
% are to be replaced by non-problematic entries. 

% fixed inds are those rows/cols that have already had jitter applied prior
% to the calling of these functions
fixed_inds = any(isnan(in))';

sqd_jitters = ~fixed_inds.*max(eps,max(in(:))).*1e-4;
out = in + diag(sqd_jitters);
return;

N = length(in);

if nargin<2 || isempty(importance)
    importance = ones(N,1);
elseif length(importance) ~= N
    error('importance must be a vector of the same length as the side length of in');
end
if nargin<3
    allowed_error = 10^-14;
end
if nargin<4
    identify_problems = false;
else
    identify_problems = strcmpi(flag, 'identify_problems');
end



% see conditioning_as_a_fn_of_number_dissim_sqdexp.m for details of these
% constants, obtained by regression.

const = -12.6398;
scale_num = -0.8594;
scale_dissim = 11.9526;

% these are the constants for non-SE matrices
% const = -17.9286;
% scale_num = 0.1879;
% scale_dissim = 6.7223;

too_similar_fn = @(n) 1-10.^(-(allowed_error * 10.^(-const) * n.^(-scale_num)).^(1/scale_dissim));
% what if all inputs were too close?
too_similar_num_in = too_similar_fn(N);

% any (non-diagonal) elements that are larger in magnitude than too_similar
% cause problems. This is a simplification.
sqd_scales = diag(in);
inv_scales = sqrt(sqd_scales).^-1;
inv_scale_mat = repmat(inv_scales,1,N);

upper_inds = triu(true(N),1);
active_inds = and(upper_inds,~isnan(in));
% fixed inds are those rows/cols that have already had jitter applied prior
% to the calling of these functions
fixed_inds = any(isnan(in));
importance(fixed_inds) = inf;

mod_in = triu(in,1);
usable_inv_scales = inv_scale_mat(active_inds);
usable_in = in(active_inds);
usable_inv_scales_t = inv_scale_mat';
usable_inv_scales_t = usable_inv_scales_t(active_inds);

mod_in(active_inds) = abs(usable_inv_scales.*usable_in.*usable_inv_scales_t);
[problem_xinds,problem_yinds]=find(mod_in>too_similar_num_in);

% a = [];
% b = [];

if identify_problems
    % don't actually modify the matrix, just report which rows and columns
    % are problematic
    out = false(N,1);
else
    sqd_jitters = zeros(N,1);
end

while ~isempty(problem_xinds) % We still have problems
    % Remove the sample with the lowest importance - if there's a tie,
    % remove the sample that lead to the most problems
    
    num_problems = histc([problem_xinds;problem_yinds],1:N);
    
%     preference_matrix=[importance(some_problems),...
%         -num_problems(some_problems)];    
    most_problems = max(num_problems);
    candidates = find(num_problems>=max(1,most_problems-2));
        
    [min_importance, min_ind] = min(importance(candidates,1));

    current_problem_pt=candidates(min_ind);

    % The indices of the problems associated with this point.
    left_problems = find(problem_xinds==current_problem_pt);
    right_problems = find(problem_yinds==current_problem_pt);
    
    current_problems=[left_problems;right_problems];

                    
    current_prob_xinds = problem_xinds(current_problems);
    current_prob_yinds = problem_yinds(current_problems);
    problems_vec = current_prob_xinds + N*(current_prob_yinds-1);

    % These are the points that current_problem_pt is too similar to
    other_pts = [problem_yinds(left_problems);...
                problem_xinds(right_problems)];
    num_problems = length(other_pts);
                
    if ~identify_problems
        % too_similar = K(x,y) / sqrt( (K(x,x) + sqd_jitter) * K(y,y) ); 
        % too_similar is what we're trying to achieve using the jitter. One
        % problem is that here we only take into account the `problem points',
        % where all other points will also influence the conditioning.
        sqd_jitters(current_problem_pt) = ...
            max(0,max(-sqd_scales(current_problem_pt) ...
            +too_similar_fn(num_problems)^(-2) * in(problems_vec).^2./...
            sqd_scales(other_pts)));
    else
        out(current_problem_pt) = true;
    end
    problem_xinds(current_problems)=[];
    problem_yinds(current_problems)=[];
    
%     a = [a;sqd_jitters(current_problem_pt)];
%     b = [b;length(problem_xinds)];

end

if ~identify_problems
    diag_sqd_jitters = diag(sqd_jitters);
    out = in + diag_sqd_jitters;
end
