function Mean = wderiv_mean_fn(hps_struct,flag)

const_ind = hps_struct.MeanConst;

if nargin<2
    flag='plain';
end

[flag, grad_hp_inds] = process_flag(flag);
switch flag
    case 'plain'
        Mean = @(hps, Xs) fMean(hps, Xs, const_ind);
    case 'grad inputs'
        Mean=@(hps, Xs) Dinputs_Mean(Xs);
    case 'hessian inputs'
        Mean=@(hps, Xs) Dinputs_Mean(Xs);
    case 'grad hyperparams'
        Mean=@(hps, Xs) Dhps_Mean(hps, Xs ,const_ind);
end

function Mean  = fMean(hps, Xs, const_ind)
if size(hps,1) == 1
    hps = hps';
end

Mean = hps(const_ind)*(Xs(:,end)==0);

function Dinputs = Dinputs_Mean(Xs)

[L,num_dims] = size(Xs);
num_dims = num_dims-1;
Dinputs = mat2cell2d(zeros(num_dims*L,L),L*ones(num_dims,1),L);

function DMean = Dhps_Mean(hps, Xs, mean_pos)

L = size(Xs,1);
num_hps = length(hps);

DMean = mat2cell2d(zeros(num_hps*L,1),L*ones(num_hps,1),1);
DMean{mean_pos} = Xs(:,end)==0;