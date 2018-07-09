function Mean = constant_mean_fn(mean_pos, flag)

if nargin<2
    flag='plain';
end

[flag, grad_hp_inds] = process_flag(flag);
% no effort to compute gradient wrt hyperparams, so do it for all anyway
switch flag
    case 'plain'
        Mean = @(hps,Xs) hps(mean_pos);
    case 'grad inputs'
        Mean=@(hps,Xs) Dinputs_Mean(Xs);
    case 'hessian inputs'
        Mean=@(hps,Xs) Dinputs_Mean(Xs);
    case 'grad hyperparams'
        Mean=@(hps,Xs) Dhps_Mean(hps, Xs, mean_pos);
    case 'sp grad inputs'
        Mean=@(hps,Xs) spDinputs_Mean(Xs);
end

function Dinputs = Dinputs_Mean(Xs)

[L,num_dims] = size(Xs);
Dinputs = mat2cell2d(zeros(num_dims*L,L),L*ones(num_dims,1),L);

function Dinputs = spDinputs_Mean(Xs)

[L,num_dims] = size(Xs);
Dinputs = zeros(L,1,num_dims);

function DMean = Dhps_Mean(hps, Xs, mean_pos)

L = size(Xs,1);
num_hps = length(hps);

DMean = mat2cell2d(zeros(num_hps*L,1),L*ones(num_hps,1),1);
DMean{mean_pos} = ones(L,1);