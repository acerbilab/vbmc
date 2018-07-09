function [K,out2] = poly_cov_fn(dim,hp,flag)       

if nargin<3
    flag = 'vector K';
end

constant_ind = 3;
constant = exp(hp(constant_ind));


K = @(x,y) (constant.^2 + (x*y')).^dim;

switch flag
    case 'vector K'
        out2 = @(x,y) (constant.^2 + sum(x.*y,2)).^dim;
    case 'deriv hyperparams'
        num_hps = length(hp);
        out2=@(Xs1,Xs2) Dhps_K(Xs1,Xs2,constant,constant_ind,num_hps);
end



function DK = Dhps_K(Xs1,Xs2,constant,constant_ind,num_hps)
L1 = size(Xs1,1);
L2 = size(Xs2,1);

DK = mat2cell2d(zeros(num_hps*L1,L2),L1*ones(num_hps,1),L2);
DK{constant_ind} = ...
    2*constant.^2*ones(L1,L2);