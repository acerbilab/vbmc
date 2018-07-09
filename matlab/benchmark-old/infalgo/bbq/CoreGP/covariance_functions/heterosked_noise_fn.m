function noise_mat = heterosked_noise_fn(noise_vec, flag)

if nargin<2
    flag='plain';
end

[flag, grad_hp_inds] = process_flag(flag);
% no effort to compute gradient wrt hyperparams, so do it for all anyway
switch flag
    case 'plain'
        noise_mat = @(hps, Xs) diag(noise_vec.^2);
    case 'grad inputs'
        noise_mat=@(hps, Xs) Dinputs_noise_mat(Xs);
    case 'hessian inputs'
        noise_mat=@(hps, Xs) Dinputs_noise_mat(Xs);
    case 'grad hyperparams'
        noise_mat=@(hps, Xs) Dhps_noise_mat(hps, Xs);
end

function Dnoise_mat = Dinputs_noise_mat(Xs)

[L,num_dims] = size(Xs);
Dnoise_mat = mat2cell2d(zeros(num_dims*L,L),L*ones(num_dims,1),L);

function Dnoise_mat = Dhps_noise_mat(hps, Xs)

L = size(Xs,1);
num_hps = length(hps);

Dnoise_mat = mat2cell2d(zeros(num_hps*L,L),L*ones(num_hps,1),L);