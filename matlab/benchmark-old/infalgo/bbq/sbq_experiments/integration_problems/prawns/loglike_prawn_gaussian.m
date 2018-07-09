function log_l_pdf = loglike_prawn_gaussian(theta, direction, model_idx)
%
% Find the log-likelihood of a setting of the nuisance parameters in a
% prawn-mind model.
%
% Reformulated from the code from Rich to use Gaussian priors on unrestricted
% domains.
%
% parameters are:
%
% * range of interaction (for spatial models)
%
% * number of neighbours to interact with (topological model)
%
% * interaction strength with prawns travelling in the opposite direction
%
% * interaction strength with prawns travelling in the same direction (for
% these, positive makes you more likely to turn around, negative less
% likely)
% 
% * decay of memory factor per timestep (1 no decay, 0 no memory)
% 
% * intensity of random turning (I basically fixed this based on 1 prawn
% experiments, otherwise it tries to assign far too much to 'randomness')

if nargin < 3; model_idx = 1; end


% downsample inputs, correlation length is ~10 frames
%for i = 1:numel(theta)
%    theta = theta(:, 1:2:end);
%    direction = direction(:, 1:2:end);
%end

switch model_idx
    case 0
        f = @logP_ring_null;
    case 1
        f = @logP_ring_mf;
    case 2
        f = @logP_ring_topo;
    case 3
        f = @logP_ring_R;
    case 4
        f = @logP_ring_R_2ways;
    case 5
        f = @logP_ring_R_ahead;
    case 6
        f = @logP_ring_R_ahead2ways;       
    case 7
        f = @logP_ring_memory;
    case 8
        f = @logP_ring_memory_2ways;     
    case 9
        f = @logP_ring_memory_ahead;
    case 10
        f = @logP_ring_memory_ahead2ways;   
end

log_l_pdf = @(x) log_l_pdf_f(f, theta, direction, x);

function log_l = log_l_pdf_f(f, theta, direction, x)

for i = 1:size(x, 1)
    K = nan; % should never be used for our problems
    R = logistic(x(i,1), 0, pi);
    p_pulse = x(i, 2:3);
    decay = logistic(x(i, 4), 0, 1);
    q = x(i, 5);

    log_l(i) = f(theta, direction, R, K, p_pulse, decay, q);
end

