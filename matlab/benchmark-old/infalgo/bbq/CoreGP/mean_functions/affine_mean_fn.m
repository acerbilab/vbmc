function Mean = affine_mean_fn(hps_struct,flag)

planar_weight_inds = hps_struct.PlanarMeanWeights;
const_ind = hps_struct.MeanConst;






if nargin<2
    flag='plain';
end

[flag, grad_hp_inds] = process_flag(flag);
switch flag
    case 'plain'
        Mean = @(hps, Xs) fMean(Xs, hps, planar_weight_inds, const_ind);
    case 'grad inputs'
        Mean=@(hps, Xs) Dinputs_Mean(Xs, hps, planar_weight_inds);
    case 'hessian inputs'
        Mean=@(hps, Xs) DDinputs_Mean(Xs);
    case 'grad hyperparams'
        Mean=@(hps, Xs) Dhps_Mean(Xs, hps,...
            planar_weight_inds,const_ind);
    case 'sp grad inputs'
        Mean=@(hps, Xs) spDinputs_Mean(Xs, hps, planar_weight_inds);
end

function Mean  = fMean(Xs, hps, planar_weight_inds, const_ind)
if size(hps,1) == 1
    hps = hps';
end

Mean = Xs*hps(planar_weight_inds) + hps(const_ind);

function DMean = Dinputs_Mean(Xs, hps, planar_weight_inds)
if size(hps,1) == 1
    hps = hps';
end
planar_weights = hps(planar_weight_inds);
[L,dims] = size(Xs);

% kron2d is faster
% tic; for i =1:1000
% v1 = repmat(planar_weights',L,1);
% v2 = v1(:);
% %v2 = kron2d(planar_weights,ones(L,1));
% end;toc

DMean = mat2cell2d(kron2d(planar_weights,ones(L,1)),L*ones(dims,1),1);

function DMean = spDinputs_Mean(Xs, hps, planar_weight_inds)
if size(hps,1) == 1
    hps = hps';
end
planar_weights = hps(planar_weight_inds);
[L,dims] = size(Xs);

planar_weights = reshape(planar_weights,1,1,dims);
DMean = repmat(planar_weights,L,1);

function DDMean = DDinputs_Mean(Xs)
% below doesn't look right, why is num_hps necessary?

% [L,dims] = size(Xs);
% DDMean = mat2cell2d(zeros(L*num_hps,num_hps),L*ones(dims,1),ones(dims,1));
% 

function DMean = Dhps_Mean(Xs, hps,...
    planar_weight_inds,const_ind)
L = size(Xs,1);
num_hps = length(hps);

DMean = mat2cell2d(zeros(num_hps*L,1),L*ones(num_hps,1),1);

DMean(planar_weight_inds) = mat2cell2d(Xs(:),L*ones(length(planar_weight_inds),1),1);
DMean{const_ind} = ones(L,1);
