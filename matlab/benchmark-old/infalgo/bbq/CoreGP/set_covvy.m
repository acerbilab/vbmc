function covvy = set_covvy(covfn_name, meanfn_name, covvy, XData, yData, num_hypersamples)
% covvy = set_covvy(covfn_name, meanfn_name, covvy, XData, yData,
% num_hypersamples)
% covfn_name can be 'sqdexp', 'matern', 'ratquad', 'poly' or 'prodcompact'
% (all of which can be read about in Rasmussen & Williams) and meanfn_name
% can be 'constant', 'planar' or 'quadratic'. The mean function's
% hyperparameters are all set by performing a least-squares fit.

if nargin<6
    num_hypersamples = 1000;
end
num_hypersamples = max(1, ceil(num_hypersamples));

if isempty(covvy)
    covvy = struct();
    num_existing_samples = 1;
else
    num_existing_samples = prod([covvy.hyperparams.NSamples]);
end

hps_struct = set_hps_struct(covvy);
if isfield(covvy,'hyperparams')
    num_existing_hps = numel(covvy.hyperparams);
else
    num_existing_hps = 0;
end

have_data = nargin >= 5;
create_logNoiseSD = ~isfield(hps_struct,'logNoiseSD');
create_logInputScales = ~isfield(hps_struct,'logInputScales') ...
                        || isempty(hps_struct.logInputScales);
create_logOutputScale = ~isfield(hps_struct,'logOutputScale')...
                        || isempty(hps_struct.logOutputScale);
create_covfn = ~isfield(covvy,'covfn')...
                        || isempty(covvy.covfn);
create_meanfn = (~isfield(covvy,'meanfn')...
                        || isempty(covvy.meanfn)) && ...
                (~isfield(covvy,'meanPos')...
                        || isempty(covvy.meanPos));


if have_data
    num_dims = size(XData,2);
    num_hps_to_create = ...
        create_logNoiseSD + num_dims*create_logInputScales + create_logOutputScale;

    num_samples = factor_in_odds(num_hypersamples/num_existing_samples,num_hps_to_create);
    output_scale = std(yData);
    input_scales = std(XData);
end

if create_logNoiseSD
    if have_data
        ind = incr_num_hps(covvy);
        
        covvy.hyperparams(ind)=struct('name','logNoiseSD',...
            'priorMean',log(0.1*output_scale),'priorSD',2,'NSamples',num_samples(ind-num_existing_hps),'type','real');
    else
        disp('Need to specify a prior for logNoiseSD, or include data to create one')
    end
end
if create_logInputScales
    if have_data
        for dim = 1:num_dims   
            ind = incr_num_hps(covvy);

            covvy.hyperparams(ind)=struct('name',['logInputScale',num2str(dim)],...
                'priorMean',log(input_scales(dim)),'priorSD',2,'NSamples',num_samples(ind-num_existing_hps),'type','real');
        end
    else
        disp('Need to specify a prior for logInputScales, or include data to create one')
    end
end
if create_logOutputScale
    if have_data
        ind = incr_num_hps(covvy);
        
        covvy.hyperparams(incr_num_hps(covvy))=struct('name','logOutputScale',...
            'priorMean',log(output_scale),'priorSD',1,'NSamples',num_samples(ind-num_existing_hps),'type','real');
    else
        disp('Need to specify a prior for logOutputScale, or include data to create one')
    end
end

if create_covfn
    hps_struct = set_hps_struct(covvy);
    % set the covariance function
    covvy.covfn = @(varargin) versatile_cov_fn(hps_struct,covfn_name,varargin{:});
end

if create_meanfn
    switch meanfn_name
        case 'constant'
            covvy = set_constant_mean(covvy, XData, yData);
        case 'planar'
            covvy = set_planar_mean(covvy, XData, yData);
        case 'quadratic'
            covvy = set_quadratic_mean(covvy, XData, yData);
        otherwise
            % assume constant.
            covvy = set_constant_mean(covvy, XData, yData);
    end
end

% if create_logInputScales
%     hps_struct = set_hps_struct(gp);
%     input_scale_inds = hps_struct.logInputScales;
%     mu = gp.meanfn(gp.hypersamples(1).hyperparameters);
%     yData_minus_mu = yData-mu(XData);
%     for dim = 1:num_dims 
%         input_scale_ind = input_scale_inds(dim);
%         [sorted_xd, sort_order] = sort(XData(:,d));
%         mean_step = mean(diff(sorted_xd)) / range(XData(:,d));
%         input_scale(d) = mean_step * output_scale / mean(abs(diff(yData_minus_mu(sort_order))))
% 
%         % rough estimate for inverse input scales
%         input_scores(d) = mean(abs(diff(yData_minus_mu(sort_order))))/mean_step;
%     end
% end

function num = incr_num_hps(covvy)
if ~isfield(covvy,'hyperparams')
    num = 1;
else
    num = numel(covvy.hyperparams)+1;
end

function factors = factor_in_odds(big_number,num_factors)
big_number = floor(big_number);
num_factors = max(1,num_factors);

if big_number<3^num_factors
    number_of_threes=floor(log(big_number)/log(3));
    factors = ones(num_factors,1);
    factors(2:(1+number_of_threes)) = 3;
else
    odd_num = floor(big_number^(1/(num_factors)));
    if odd_num/2 == floor(odd_num/2)
        odd_num = odd_num-1;
    end
    factors = ones(num_factors,1) * odd_num;
    ind = 1; % this is deliberate, we want to add samples to input scales before the noise
    while prod(factors)<big_number
        ind = ind+1;
        if ind > num_factors
            ind = 1;
        end
        factors(ind) = odd_num;
        if ind == 1
            odd_num = odd_num+2;
        end
    end
    factors(ind) = factors(ind)-2;
end
