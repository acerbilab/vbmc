function [K,out2,out3] = versatile_cov_fn(hps_struct,type,hp,flag)       



if ischar(type) || iscell(type)
    % inputs are as hps_struct,type,hp,flag
    
    num_hps = length(hp);
    
    input_scale_inds = hps_struct.logInputScales;
    output_scale_ind = hps_struct.logOutputScale;
    
    if nargin<4
        flag='deriv inputs';
    end
else
    % inputs are as type,hp,flag -- this option exists for
    % back-compatibility
    if nargin<3
        flag='deriv inputs';
    else
        flag = hp;
    end
    
    hp = type;
    type = hps_struct;
    
    num_hps = length(hp);
        
    input_scale_inds = 3:(num_hps-1);
    output_scale_ind = num_hps;
end



T=exp(hp(input_scale_inds));
L=exp(hp(output_scale_ind));



switch flag
    case 'deriv inputs'
        K=@(Xs1,Xs2) matrify(@(varargin) fcov(type,{T,L},varargin{:}),Xs1,Xs2);
        DK=@(Xs1,Xs2) matrify(@(varargin) gcov(type,{T,L},varargin{:}),Xs1,Xs2);
        DDK=@(Xs1,Xs2) matrify(@(varargin) Hcov(type,{T,L},varargin{:}),Xs1,Xs2);
        out2=DK;
        out3=DDK;
    case 'deriv hyperparams'
        K=@(Xs1,Xs2) matrify(@(varargin) fcov(type,{T,L},varargin{:}),Xs1,Xs2);
        out2=@(Xs1,Xs2) Dhps_K(Xs1,Xs2,type,{T,L},input_scale_inds,output_scale_ind,num_hps);
    case 'vector'
        K = @(Xs1,Xs2) fcov(type,{T,L},Xs1,Xs2,'vector');
        % not enabled
%         DK=@(Xs1,Xs2) gcov(type,{T,L},Xs1,Xs2,'vector');
%         DDK=@(Xs1,Xs2) Hcov(type,{T,L},Xs1,Xs2,'vector');
%         out2=DK;
%         out3=DDK;
end

function DK = Dhps_K(Xs1,Xs2,type,TL,input_scale_inds,output_scale_ind,num_hps)
[L1,num_dims] = size(Xs1);
L2 = size(Xs2,1);
grad_hp_inds = 1:num_dims;

DK = mat2cell2d(zeros(num_hps*L1,L2),L1*ones(num_hps,1),L2);
DK(input_scale_inds) = ...
    matrify(@(varargin) gTcov(type,TL,grad_hp_inds,varargin{:}),Xs1,Xs2);
DK{output_scale_ind} = ...
    2*matrify(@(varargin) fcov(type,TL,varargin{:}),Xs1,Xs2);
