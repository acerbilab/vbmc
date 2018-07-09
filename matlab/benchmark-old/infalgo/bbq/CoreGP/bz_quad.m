function output = bz_quad(prior, varargin)
% output = bz_quad(prior, varargin)
% Returns the mean estimate for an integral. 
% prior is a structure specifying the first term in the integral. 
% varargin are the remaining terms in the integral, assumed to be of the
% form of a GP conditional mean. We assume the covariance for each of these
% GPs is Gaussian (normalised sqd exp).

num_terms = length(varargin);
num_variables = numel(prior);

tau = cell(1,num_terms);
sample_locations = cell(1,num_terms);
widths = nan(num_terms, num_variables);
for term = 1:num_terms
    GP = varargin{term};
    ML_ind = max(GP.hypersamples.logL);
    ML_hypersample = GP.hypersamples(ML_ind).hyperparameters;
    
    is_input_scale_cell = strfind(names,'logInputScale');
    input_scale_inds = ~cellfun(@(x) isempty(x),is_input_scale_cell);
    
    % tau is inv(K(variable_samples,variable_samples))*f(variable_samples)
    tau{term} = GP.invKData;
    sample_locations{term} = GP.xd;
    widths(term,:) = ML_hypersample(input_scale_inds);
end

% need some test for if there are derivative observations (for which
% GP.invKdata will already have used the correct covariance).


Yotta = nan(cellfun(@length,sample_locations));
for variable = 1:num_variables
    type=prior(variable).type;

    priorMean=prior(input).mean;
    priorSD=prior(input).SD;

    if ~all(~isnan([priorMean;priorSD]));
        % This variable is a dummy - ignore it
        continue
    end

    switch lower(type)
        case 'real'
            switch num_terms
                case 1
                    
                case 2
                    meshgrid
            end
        case 'bounded'

        case 'discrete'
        
        case 'mixture'
            mixtureWeights = prior(input).mixtureWeights;

            ns = 0;
            for i = 1:length(mixtureWeights)
                ns = ns + mixtureWeights(i)*normpdf(samples_input,priorMean(i),sqrt(priorSD(i)^2+width^2));
            end
    end

    Yotta = Yotta .* Yotta_var;
end

output = Yotta;
% tprod is faster than ttv, indeed, it is almost as fast as matrix
% multiplication where comparable.
for term = 1:num_terms
    output = tprod(output,[-1,1:(num_terms-term)],tau{term},-1);
end
