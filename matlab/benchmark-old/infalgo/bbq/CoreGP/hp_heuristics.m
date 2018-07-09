function [est_noise_sd,est_input_scales,est_output_scale] = ...
    hp_heuristics(XData,yData_minus_mu,num_samples)
% [est_noise_sd,est_input_scales,est_output_scale] = ...
%     hp_heuristics(XData,yData_minus_mu,num_samples)
% yData_minus_mu is the data with the appropriate mean function already
% subtracted off

if nargin<3
    num_samples = 120;
end

num_dims = size(XData,2);
num_data = size(XData,1);

if num_data == 1
    est_input_scales = ones(1,num_dims);
    est_output_scale = max(1,yData_minus_mu);
    est_noise_sd = 0.1*yData_minus_mu;
    return
elseif num_data > 10000
    filter = ceil(linspace(1, num_data, 10000));
    XData = XData(filter,:);
    yData_minus_mu = yData_minus_mu(filter,:);
    num_data = size(XData,1);
end


% window length to use for 
noise_length = min(num_data,max(5,ceil(num_data/50)));
dist_step = ceil(num_data/50);

yData_minus_mu(yData_minus_mu == -Inf) = -500;

% a and b, parameters of the Gamma prior for the precision should be
% essentially washed out by sufficient data
a = 1;
b = 1e-3;

% input scale for BQ over correlation constant R
width = 0.01;
% samples for BQ over correlation constant R
R_vec = exp(-0.5*(exp(linspace(-3,3,num_samples))).^2)';
num_R = length(R_vec);

est_noise_sd = nan(1,num_dims);
est_output_scale = nan(1,num_dims);
est_input_scales = nan(1,num_dims);

for dim = 1:num_dims
    [XData_d, sort_order] = sort(XData(:,dim));
    mean_step = mean(diff(XData_d));

    yData_minus_mu_d = yData_minus_mu(sort_order);

    num = floor(noise_length/2);
    included_pts = (num+1):(num_data-num);

    mat = repmat(yData_minus_mu_d,1,num_data);
    filtered = ...
        sum(triu(mat,-num)-triu(mat,-num+noise_length))/noise_length;  
    filtered = filtered(included_pts)';

    est_noise_sd(dim) = ...
        std(yData_minus_mu_d(included_pts,:) - filtered);
    yData_minus_mu_noise_d = filtered;




    

    
    y1 = yData_minus_mu_noise_d(1:end-dist_step+1);
    y2 = yData_minus_mu_noise_d(dist_step:end);
    num_y = length(y1);

    sum_sqd_y1_y2 = sum(y1.^2) + sum(y2.^2);
    sum_y1y2 = sum(y1.*y2);
    
    
    term = b + 0.5./(1-R_vec.^2) * sum_sqd_y1_y2 ...
        - R_vec./(1-R_vec.^2) * sum_y1y2;
    log_r = (-num_y-a).*log(term);
    q = term.^(-1)*(num_y+a);

    Nfn = @(r1,r2) normpdf(r1,r2,sqrt(2)*width).*...
            (normcdf(1,0.5*(r1+r2),(1/sqrt(2))*width) - ...
            normcdf(0,0.5*(r1+r2),(1/sqrt(2))*width));
    N = matrify(Nfn, R_vec, R_vec);

    log_r = log_r - max(log_r);
    r = exp(log_r);

    eta = normcdf(1,R_vec,width) - normcdf(0,R_vec,width);
    m_eta = R_vec.*eta - width^2*...
        (normpdf(1,R_vec,width) - normpdf(0,R_vec,width));
    K = improve_covariance_conditioning(...
        matrify(@(x,y) normpdf(x,y,width), R_vec, R_vec), r);

    cholK = chol(K);
    invK_r = solve_chol(cholK,r);
    denom = eta'*invK_r;

    mean_R = (m_eta' * invK_r) / denom;
    mean_prec = (solve_chol(cholK,q)'*N*invK_r) / denom;

    est_input_scales(dim) = dist_step*mean_step/(sqrt(-2*log(mean_R)));
    est_output_scale(dim) = sqrt(mean_prec^-1);

end

% % over-estimating the input scales is usually better than under-estimating
% est_input_scales = 10*est_input_scales;

est_input_scales = min(est_input_scales, 3*range(XData));

problems = or(or(...
            isnan(est_input_scales), ...
            (abs(real(est_input_scales)-est_input_scales) > eps)), ...
            range(XData)<eps);
        


est_input_scales(problems) = 1;
est_output_scale(isnan(est_output_scale)) = max(yData_minus_mu);
est_output_scale(isinf(est_output_scale)) = max(yData_minus_mu);
est_output_scale(est_output_scale<eps) = max(yData_minus_mu);
est_output_scale = max(est_output_scale);
est_noise_sd = max(eps,min(est_noise_sd));


