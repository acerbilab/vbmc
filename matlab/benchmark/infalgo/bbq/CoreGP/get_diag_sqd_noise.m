function noise_fun = get_diag_sqd_noise(gp, flag)
% usgae:
% diag_noise = get_diag_noise(gp, 'plain');
% if grad_hyperparams
%     Ddiag_noise = get_diag_noise(gp, {'grad hyperparams',grad_hp_inds});
% end

if nargin<2
    flag = 'plain';
end

noise_fun_test = isfield(gp,'noisefn');
if ~noise_fun_test 
    if (~isfield(gp, 'logNoiseSDPos'))
        names = {gp.hyperparams.name};
        logNoiseSDPos = cellfun(@(x) strcmpi(x, 'logNoiseSD'), names);
    else
        logNoiseSDPos = gp.logNoiseSDPos;
    end
    
    % if logNoiseSDPos is empty, this means we assume zero noise.
    if ~isempty(logNoiseSDPos)
        noise_fun = diag_sqd_iid_noise_fn(logNoiseSDPos, flag);
    else
        noise_fun = @(X,hps) 0; % no matter the flag
    end
else
    noise_fun = gp.noisefn(flag);
end

function sqd_noise_vec = diag_sqd_iid_noise_fn(logNoiseSDPos, flag)

if nargin<2
    flag='plain';
end

[flag, grad_hp_inds] = process_flag(flag);
% no effort to compute gradient wrt hyperparams, so do it for all anyway

switch flag
    case 'plain'
        sqd_noise_vec = @(hps, Xs) ...
            exp(hps(logNoiseSDPos))^2*ones(1,size(Xs,1));
%     case 'grad inputs'
%         sqd_noise_vec=@(hps, Xs) Dinputs_noise_vec...
%             (hps, Xs, logNoiseSDPos, grad_hp_inds);
%     case 'hessian inputs'
%         sqd_noise_vec=@(hps, Xs) Dinputs_noise_vec...
%             (hps, Xs, logNoiseSDPos, grad_hp_inds);
    case 'grad hyperparams'
        sqd_noise_vec=@(hps, Xs) Dhps_noise_mat...
                            (hps, Xs, logNoiseSDPos, grad_hp_inds);
end

function Dnoise_mat = Dhps_noise_mat(hps, Xs, logNoiseSDPos, grad_hp_inds)

num_data = size(Xs,1);
[is_noise_ind, pre_ind] = ismember(logNoiseSDPos, grad_hp_inds);
num_derivs = length(grad_hp_inds);

Dnoise_mat = zeros(1, num_data, num_derivs);

if is_noise_ind
    % we actually want the derivative wrt log(noise)
    Dnoise_mat(:, :, pre_ind) = 2 * exp(hps(logNoiseSDPos))^2 * ...
        ones(num_data,1);
end