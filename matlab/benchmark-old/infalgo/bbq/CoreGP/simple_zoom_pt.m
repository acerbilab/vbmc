function [x_zoomed, local_optimum_flag, y_zoomed] = simple_zoom_pt(x, grad, input_scales, min_max_local_optimum_flag)
% find the approximate location of the local minimum near to x. We assume a
% constant mean function, Gaussian covariance and noiseless observations.
% x: point/s to `zoom' from, num_pts * num_dims
% grad: gradient at x
% input_scale: input scale for gp over f(x), can be either 1 * num_dims or
%               num_pts * num_dims
% best_ind: hyperparameter sample index to use
% x_zoomed: location of local minimum `zoomed' to
% y_zoomed: value of local minimum `zoomed' to (assuming the value at x was
%               zero)
% local_optimum_flag: point unmoved, likely a local minimum
% 
% NB: due to the assumed covariance, a product of independent terms over
% each input dimension, the zoomed_x will not, in general, be shifted by a
% multiple of the grad.*input_scales from x; the input scales being
% identical giving an exception.

inv_sqd_input_scales = input_scales.^-2;

num_pts = size(x,1);
num_dims = size(x,2);

if size(grad,2) ~= num_dims
    grad = grad';
end

if nargin<4
    min_max_local_optimum_flag = 'minimise';
end

x(x == -inf) = -2^1023;
x(x == inf) = 2^1023;

diag_Kmat = [ones(1,size(inv_sqd_input_scales,1));inv_sqd_input_scales'];
% scale by taking mean = f(x); we only want to find the location after all
full_alphas = bsxfun(@rdivide,[zeros(1,num_pts);grad'],diag_Kmat)';
alpha0 = full_alphas(:,1);
alphas = full_alphas(:,2:end);

const_sum = sum(bsxfun(@times, alphas.^2, inv_sqd_input_scales), 2);
local_optimum_flag = const_sum < eps;

x_zoomed = x;
y_zoomed = zeros(num_pts,1);

if all(local_optimum_flag)
    local_optimum_flag = all(local_optimum_flag);
    return
end

full_alphas = full_alphas(~local_optimum_flag,:);
alpha0 = alpha0(~local_optimum_flag,:);
alphas = alphas(~local_optimum_flag,:);
const_sum = const_sum(~local_optimum_flag,:);
x = x(~local_optimum_flag,:);
y = y_zoomed(~local_optimum_flag,:);
if size(inv_sqd_input_scales,1) == num_pts
    inv_sqd_input_scales = inv_sqd_input_scales(~local_optimum_flag,:);
end

const_term=sqrt(alpha0.^2+4*const_sum);

% two possible solutions, we'll test them both and return the better
const_left=(-alpha0 - const_term)./(2*const_sum);
const_right=(-alpha0 + const_term)./(2*const_sum);

x_left = bsxfun(@times, const_left, alphas) + x;
x_right = bsxfun(@times, const_right, alphas) + x;

K = exp(-0.5*sum(bsxfun(@times, (x_left-x).^2, inv_sqd_input_scales),2));
DK = bsxfun(@times, bsxfun(@times, (x_left-x), inv_sqd_input_scales), K);
K_DK = [K, DK];
mu_left = sum(K_DK.*full_alphas,2);

K = exp(-0.5*sum(bsxfun(@times, (x_right-x).^2, inv_sqd_input_scales),2));
DK = bsxfun(@times, bsxfun(@times, (x_right-x), inv_sqd_input_scales), K);
K_DK = [K, DK];
mu_right = sum(K_DK.*full_alphas,2);

if strcmpi(min_max_local_optimum_flag,'minimise')
    x = x_right;
    y = mu_right;
    
    left_inds = mu_left < mu_right;
    
    x(left_inds, :) = x_left(left_inds, :); 
    y(left_inds, :) = mu_left(left_inds, :);
elseif strcmpi(min_max_local_optimum_flag,'maximise')
    x = x_right;
    y = mu_right;
    
    left_inds = mu_left > mu_right;
    
    x(left_inds, :) = x_left(left_inds, :); 
    y(left_inds, :) = mu_left(left_inds, :);
end

x_zoomed(~local_optimum_flag, :) = x;
y_zoomed(~local_optimum_flag) = y;