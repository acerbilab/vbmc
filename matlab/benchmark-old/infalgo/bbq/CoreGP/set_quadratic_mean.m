function covvy = set_quadratic_mean(covvy, XData, YData, num_samples)
% Set up covvy with a quadratic prior mean. It can be called in three ways:
% covvy = set_quadratic_mean(covvy, num_samples) sets zero mean, 1 SD
% priors for each hyperparameter of the mean function, with num_samples
% taken from each. 
% covvy = set_quadratic_mean(covvy, XData, YData) 
% takes a least-squares fit for the supplied data and sets appropriate
% delta function priors for each hyperparameter of the mean function. 
% covvy = set_quadratic_mean(covvy, XData, YData, num_samples) 
% takes a least-squares fit for the supplied data and uses it to give the
% means for the priors for each hyperparameter of the mean function. The SD
% taken for each is also estimated from the least-squares fit.

hps_struct = set_hps_struct(covvy);
num_dims = hps_struct.num_dims;

    [a1,b1] = meshgrid(1:num_dims,1:num_dims);
    a2 = tril(a1);
    b2 = tril(b1);
    a3 = a2(:);
    b3 = b2(:);
    a3(a3==0) = [];
    b3(b3==0) = [];
    % see nchoosek

num_quad_weights = length(a3);

if nargin == 2
    num_samples = XData;
    type = 'real';
    
    quad_weights = zeros(num_quad_weights,1);
    quad_priorSDs = ones(num_quad_weights,1);
    planar_weights = zeros(num_dims,1);
    planar_priorSDs = 10*ones(num_dims,1);
    const = 0;
    const_priorSD = 10;
elseif nargin>2
    [num_Data] = size(XData,1);
    
    trial_terms = [XData(:,a3).*XData(:,b3) XData ones(num_Data,1)];
    num_terms = size(trial_terms,2);
    least_squares_soln = trial_terms\YData;
    quad_weights = least_squares_soln(1:num_quad_weights);
    planar_weights = least_squares_soln(num_quad_weights+(1:num_dims));
    const = least_squares_soln(end);
    


    if nargin>3
        type = 'real';
        
        error = @(x) sqrt(sum((YData - trial_terms*x).^2));
        best_error = error(least_squares_soln);
        objective = @(x) abs(best_error*exp(1)-error(x));
        upper_soln = nan(num_terms,1);
        for i=1:num_terms
            objective_i = @(xi) ...
                objective([least_squares_soln(1:i-1);
                            xi;least_squares_soln(i+1:end)]);
            upper_soln(i) = fsolve(objective_i,least_squares_soln(i));
        end

        SDs = abs(upper_soln - least_squares_soln);

        quad_priorSDs = SDs(1:num_quad_weights);
        planar_priorSDs = SDs(num_quad_weights+(1:num_dims));
        const_priorSD = SDs(end);
    else
        num_samples = 1;
        type = 'inactive';
        
        quad_priorSDs = ones(num_quad_weights,1);
        planar_priorSDs = ones(num_dims,1);
        const_priorSD = 1;
    end
end

for ind=1:length(quad_weights)
    covvy.hyperparams(numel(covvy.hyperparams)+1)=struct('name',['QuadMeanWeight',num2str(ind)],...
        'priorMean',quad_weights(ind),'priorSD',quad_priorSDs(ind),'NSamples',num_samples,'type',type);
end
for ind=1:length(planar_weights)
    covvy.hyperparams(numel(covvy.hyperparams)+1)=struct('name',['PlanarMeanWeight',num2str(ind)],...
        'priorMean',planar_weights(ind),'priorSD',planar_priorSDs(ind),'NSamples',num_samples,'type',type);
end
covvy.hyperparams(numel(covvy.hyperparams)+1)=struct('name','MeanConst',...
    'priorMean',const,'priorSD',const_priorSD,'NSamples',num_samples,'type',type);

hps_struct = set_hps_struct(covvy);
covvy.meanfn = @(varargin) quadratic_mean_fn(hps_struct,varargin{:});