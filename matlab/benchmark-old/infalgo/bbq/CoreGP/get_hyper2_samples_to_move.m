function [toMove,covvy] = get_hyper2_samples_to_move(covvy)

if ~isfield(covvy,'manage_h2s_method')
    method = 'all'; 
else
    method = covvy.manage_h2s_method;
end

switch (lower(method))
 case 'rotating'
     
    num2move = 3;
	if (isfield(covvy, 'lastHyper2SamplesMoved')) && ~isempty(covvy.lastHyper2SamplesMoved)
		lastMoved = covvy.lastHyper2SamplesMoved(1);
	else
		lastMoved = numel(covvy.hyper2samples);
	end
	
	% cycle through the hyper2 samples one at a time
	toMove = mod(lastMoved+(0:num2move-1), numel(covvy.hyper2samples)) + 1;
case 'all'
    toMove = 1:numel(covvy.hyper2samples);
case 'optimal'
    % This is necessary to get gradients for all the hyper2samples
    toMove = 1:numel(covvy.hyper2samples);
 otherwise
	toMove = [];
end
