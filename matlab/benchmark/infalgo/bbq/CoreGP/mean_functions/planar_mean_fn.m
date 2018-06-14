function [Mean, DMean, DDMean] = planar_mean_fn(hps_struct,hps,flag)

num_hps = length(hps);

planar_weight_inds = hps_struct.PlanarMeanWeights;
const_ind = hps_struct.MeanConst;

const = hps(const_ind);
planar_weights = hps(planar_weight_inds);

if size(planar_weights,1) == 1
    planar_weights = planar_weights';
end
 
Mean=@(Xs) Xs*planar_weights + const;

if nargin<3
    flag='deriv inputs';
end

if strcmpi(flag,'deriv inputs')
    DMean=@(Xs) Dinputs_Mean(Xs,planar_weights);
    DDMean=@DDinputs_Mean;
elseif strcmpi(flag,'deriv hyperparams')
    DMean=@(Xs) Dhps_Mean(Xs,...
        planar_weight_inds,const_ind,num_hps);
end

function DMean = Dinputs_Mean(Xs,planar_weights)
[L,dims] = size(Xs);

% kron2d is faster
% tic; for i =1:1000
% v1 = repmat(planar_weights',L,1);
% v2 = v1(:);
% %v2 = kron2d(planar_weights,ones(L,1));
% end;toc

DMean = mat2cell2d(kron2d(planar_weights,ones(L,1)),L*ones(dims,1),1);

function DDMean = DDinputs_Mean(Xs)
[L,dims] = size(Xs);
DDMean = mat2cell2d(zeros(L*num_hps,num_hps),L*ones(dims,1),ones(dims,1));


function DMean = Dhps_Mean(Xs,...
    planar_weight_inds,const_ind,num_hps)
L = size(Xs,1);

DMean = mat2cell2d(zeros(num_hps*L,1),L*ones(num_hps,1),1);

DMean(planar_weight_inds) = mat2cell2d(Xs(:),L*ones(length(planar_weight_inds),1),1);
DMean{const_ind} = ones(L,1);
