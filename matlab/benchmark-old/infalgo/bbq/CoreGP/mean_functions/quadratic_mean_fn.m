function [Mean, DMean, DDMean] = quadratic_mean_fn(hps_struct,hps,flag)

num_hps = length(hps);
num_dims = hps_struct.num_dims;

quad_weight_inds = hps_struct.QuadMeanWeights;
planar_weight_inds = hps_struct.PlanarMeanWeights;
const_ind = hps_struct.MeanConst;

const = (hps(const_ind));
planar_weights = (hps(planar_weight_inds));
quad_weights = (hps(quad_weight_inds));

if size(quad_weights,1) == 1
    quad_weights = quad_weights';
end

if size(planar_weights,1) == 1
    planar_weights = planar_weights';
end

[a1,b1] = meshgrid(1:num_dims,1:num_dims);
a2 = tril(a1);
b2 = tril(b1);
a3 = a2(:);
b3 = b2(:);
a3(a3==0) = [];
b3(b3==0) = [];
% see nchoosek
 
Mean=@(Xs) (Xs(:,a3).*Xs(:,b3))*quad_weights+Xs*planar_weights+const;

if nargin<3
    flag='deriv inputs';
end

if strcmpi(flag,'deriv inputs')

    Dquad_weights = nan(num_dims);
    for ind = 1:num_dims
        Dquad_weights(:,ind) = quad_weights(or(a3 == ind,b3 == ind));
    end
    Dquad_weights = Dquad_weights + diag(diag(Dquad_weights));
    Dquad_weights_cell = mat2cell2d(Dquad_weights(:),num_dims*ones(num_dims,1),1);
    
    DMean=@(Xs) Dinputs_Mean(Xs,Dquad_weights_cell,planar_weights);
    DDMean=@(Xs) DDinputs_Mean(Xs,Dquad_weights);
elseif strcmpi(flag,'deriv hyperparams')
    DMean=@(Xs) Dhps_Mean(Xs,a3,b3,...
        quad_weight_inds,planar_weight_inds,const_ind,num_hps);
end


function DMean = Dinputs_Mean(Xs,Dquad_weights_cell,planar_weights)
[L,num_dims] = size(Xs);

DMean = cellfun(@(x,y,z) x*y+z,...
    mat2cell2d(repmat(Xs,num_dims,1),L*ones(num_dims,1),1),...
    Dquad_weights_cell,...
    mat2cell2d(kron2d(planar_weights,ones(L,1)),L*ones(num_dims,1),1));

function DDMean = DDinputs_Mean(Xs,Dquad_weights)
[L,dims] = size(Xs);
DDMean = mat2cell2d(kron2d(Dquad_weights,ones(L,1)),...
    L*ones(dims,1),ones(dims,1));


function DMean = Dhps_Mean(Xs,a3,b3,...
    quad_weight_inds,planar_weight_inds,const_ind,num_hps)
L = size(Xs,1);

DMean = mat2cell2d(zeros(num_hps*L,1),L*ones(num_hps,1),1);

mat = Xs(:,a3).*Xs(:,b3);
DMean(quad_weight_inds) = mat2cell2d(mat(:),L*ones(length(quad_weight_inds),1),1);
DMean(planar_weight_inds) = mat2cell2d(Xs(:),L*ones(length(planar_weight_inds),1),1);
DMean{const_ind} = ones(L,1);
