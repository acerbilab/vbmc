function K = hom_cov_fn(hps_struct,type,flag)     
% usage: covvy.covfn = @(flag) hom_cov_fn(hps_struct,type,flag);

num_hps = hps_struct.num_hps;

input_scale_inds = hps_struct.logInputScales;
output_scale_ind = hps_struct.logOutputScale;

if nargin<3
    flag='plain';
end

[flag, grad_hp_inds] = process_flag(flag);
switch flag
    case 'plain'
        K=@(hp,varargin) matrify(@(varargin) fcov(type,...
            {exp(hp(input_scale_inds)),exp(hp(output_scale_ind))},...
            varargin{:}),varargin{:});
    case 'grad inputs'
        K=@(hp, varargin) matrify(@(varargin) gcov(type,...
            {exp(hp(input_scale_inds)),exp(hp(output_scale_ind))},...
            varargin{:}),varargin{:});
    case 'hessian inputs'
        K=@(hp, varargin) matrify(@(varargin) Hcov(type,...
            {exp(hp(input_scale_inds)),exp(hp(output_scale_ind))},...
            varargin{:}),varargin{:});
    case 'grad hyperparams'
        K=@(hp, varargin) dhps_fcov(type,...
            {exp(hp(input_scale_inds)),exp(hp(output_scale_ind))},...
            input_scale_inds,output_scale_ind,num_hps,grad_hp_inds,...
            varargin{:});
    case 'grad hyperparams grad inputs'
        K=@(hp, varargin) dhps_gcov(type,...
            {exp(hp(input_scale_inds)),exp(hp(output_scale_ind))},...
            input_scale_inds,output_scale_ind,num_hps,grad_hp_inds,...
            varargin{:});
    case 'grad hyperparams hessian inputs'
        K=@(hp, varargin) dhps_Hcov(type,...
            {exp(hp(input_scale_inds)),exp(hp(output_scale_ind))},...
            input_scale_inds,output_scale_ind,num_hps,grad_hp_inds,...
            varargin{:});
    case 'vector'
        % this can't be used in sqd_diffs mode
        K = @(hp, Xs1, Xs2) fcov(type,...
            {exp(hp(input_scale_inds)),exp(hp(output_scale_ind))},...
            Xs1, Xs2,'vector');
end

function K = dhps_fcov(type,TL,input_scale_inds,output_scale_ind,...
    num_hps,grad_hp_inds,varargin)

switch length(varargin)
    case 1
        L1 = size(varargin{1},1);
        L2 = size(varargin{1},2);
    case 2
        L1 = size(varargin{1},1);
        L2 = size(varargin{2},1);
end

K = mat2cell2d(zeros(num_hps*L1,L2),L1*ones(num_hps,1),L2);

[input_scale_inds, rel_input_scale_inds] = ...
    intersect(input_scale_inds, grad_hp_inds);
if ~isempty(input_scale_inds)
    K(input_scale_inds) = ...
        matrify(@(varargin) gTcov(type,TL,rel_input_scale_inds,...
                            varargin{:}),varargin{:});
end
 
if ismember(output_scale_ind, grad_hp_inds)
    K{output_scale_ind} = ...
        2*matrify(@(varargin) fcov(type,TL,varargin{:}),varargin{:});
end

function K = dhps_gcov(type,TL,input_scale_inds,output_scale_ind,...
    num_hps,grad_hp_inds,varargin)

switch length(varargin)
    case 1
        L1 = size(varargin{1},1);
        L2 = size(varargin{1},2);
    case 2
        L1 = size(varargin{1},1);
        L2 = size(varargin{2},1);
end

K = mat2cell2d(zeros(num_hps*L1,L2),L1*ones(num_hps,1),L2);

[input_scale_inds, rel_input_scale_inds] = ...
    intersect(input_scale_inds, grad_hp_inds);
if ~isempty(input_scale_inds)
    K(input_scale_inds) = ...
        matrify(@(varargin) gTgcov(type,TL,rel_input_scale_inds,...
                            varargin{:}),varargin{:});
end
 
if ismember(output_scale_ind, grad_hp_inds)
    K{output_scale_ind} = ...
        2*matrify(@(varargin) gcov(type,TL,varargin{:}),varargin{:});
end

function K = dhps_Hcov(type,TL,input_scale_inds,output_scale_ind,...
    num_hps,grad_hp_inds,varargin)

switch length(varargin)
    case 1
        L1 = size(varargin{1},1);
        L2 = size(varargin{1},2);
    case 2
        L1 = size(varargin{1},1);
        L2 = size(varargin{2},1);
end

K = mat2cell2d(zeros(num_hps*L1,L2),L1*ones(num_hps,1),L2);

[input_scale_inds, rel_input_scale_inds] = ...
    intersect(input_scale_inds, grad_hp_inds);
if ~isempty(input_scale_inds)
    K(input_scale_inds) = ...
        matrify(@(varargin) gTHcov(type,TL,rel_input_scale_inds,...
                            varargin{:}),varargin{:});
end
 
if ismember(output_scale_ind, grad_hp_inds)
    K{output_scale_ind} = ...
        2*matrify(@(varargin) Hcov(type,TL,varargin{:}),varargin{:});
end
