function [K] = shipping_cov_fn(hps_struct,type,flag)  
    
num_hps = hps_struct.num_hps;

input_scale_ind = hps_struct.logInputScales;
output_scale_ind = hps_struct.logOutputScale;
corr_ind = hps_struct.CorrelationAngle;

if nargin<3
    flag='plain';
end

switch flag
    case 'plain'
        K=@(hps, as, bs) matrify(@(al,ax,ay,bl,bx,by)...
            fcov({type,'great-circle'},...
            {hps(input_scale_ind),hps(output_scale_ind)},ax,ay,bx,by)...
            .*SensorCov(hps(corr_ind), al, bl),as,bs);
    case 'grad hyperparams'
        K=@(hp, varargin) dhps_fcov(type,...
            {exp(hp(input_scale_inds)),exp(hp(output_scale_ind))},...
            input_scale_inds,output_scale_ind,num_hps,grad_hp_inds,...
            varargin{:});
    case 'vector'
        % this can't be used in sqd_diffs mode
        K = @(hp, Xs1, Xs2) fcov({type,'great-circle'},...
            {hp(input_scale_ind),hp(output_scale_ind)},...
            Xs1(:,2:3), Xs2(:,2:3),'vector')...
            .*SensorCov(hp(corr_ind), Xs1(:,1), Xs2(:,1));
end
        

function K = SensorCov(theta, al, bl)

CovMat= [1, cos(theta);cos(theta), 1];
K = CovMat((bl-1)*2+al);

function K = dhps_fcov(type,TL,theta,...
    input_scale_inds,output_scale_ind,corr_ind,...
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

CovMat= [1, cos(theta);cos(theta), 1];
dCovMat= [0, -sin(theta);-sin(theta), 0];

[input_scale_inds, rel_input_scale_inds] = ...
    intersect(input_scale_inds, grad_hp_inds);
if ~isempty(input_scale_inds)
    K(input_scale_inds) = ...
        matrify(@(al,ax,ay,bl,bx,by) gTcov({type,'great-circle'},...
                            TL,rel_input_scale_inds,...
                            ax,ay,bx,by).*CovMat((bl-1)*2+al),...
                            varargin{:});
                        
end

if ismember(corr_ind, grad_hp_inds)
    K{corr_ind} = ...
        matrify(@(al,ax,ay,bl,bx,by) fcov({type,'great-circle'},...
                            TL,rel_input_scale_inds,...
                            ax,ay,bx,by).*dCovMat((bl-1)*2+al),...
                            varargin{:});
end
 
if ismember(output_scale_ind, grad_hp_inds)
    K{output_scale_ind} = ...
        2*matrify(@(al,ax,ay,bl,bx,by) fcov({type,'great-circle'},...
                            TL,rel_input_scale_inds,...
                            ax,ay,bx,by).*CovMat((bl-1)*2+al),...
                            varargin{:});
end


