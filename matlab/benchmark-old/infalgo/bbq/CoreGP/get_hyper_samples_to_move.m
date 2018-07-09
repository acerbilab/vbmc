function toMove = get_hyper_samples_to_move(covvy, method)

% Should also be a clause in here to move hypersamples that are stuffing up
% the conditioning

if (nargin < 2); method = 'rotating'; end

switch (lower(method))
    case 'rotating'
        if (isfield(covvy, 'current_sample')) && ~isempty(covvy.current_sample)
            current_sample = covvy.current_sample(end);
        else
            current_sample = numel(covvy.hypersamples);
        end

        % cycle through the hyper2 samples one at a time
        toMove = mod(current_sample, numel(covvy.hypersamples)) + 1;
    
    case 'replace_worst'
        if (isfield(covvy, 'rho'))
            [minrho,toMove]=min(covvy.rho);
        else
            toMove=1;
        end
        
    case 'replace_lowest_logl'
        if (isfield(covvy.hypersamples(1), 'logL')) && ~isempty(covvy.hypersamples(1).logL);
            [minlogL,toMove]=min([covvy.hypersamples.logL]);
        else
            toMove=1;
        end

    case 'all'
        toMove = 1:numel(covvy.hypersamples);
        
  otherwise
        toMove = [];
end
