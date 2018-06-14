function [K,out2] = sets_cov_fn(hps_struct,type,hp,flag)  
% sets_cov_fn will feed fcov the correctly partitioned inputs (that is,
% grouped into set elements) and 

    % the weights should already be stored in XData. 
    
num_hps = length(hp);
input_scale_inds = hps_struct.logInputScales;
set_input_scale_ind = hps_struct.logSetInputScale;
input_scales=exp(hp([input_scale_inds,set_input_scale_ind]));

output_scale_inds = hps_struct.logOutputScale;
output_scale = exp(hp(output_scale_inds));

% flag is redundant for now -- can't compute derivatives wrt emd

if iscell(type)
    time_type = type{2};
    type = type{1};
    
    time_scale_ind = hps_struct.logTimeScale;
    time_scale = exp(hp(time_scale_ind));
    
    K=@(as,bs) matrify(@(as,at,bs,bt)...
        sets_cov_fn_x({type,'sets'},{input_scales,output_scale},as,bs).*...
        fcov(time_type,{time_scale,1},at,bt);
else
    K=@(as,bs) matrify(@(as,bs)...
        sets_cov_fn_x({type,'sets'},{input_scales,output_scale},as,bs);
    
end
    


                
function K = sets_cov_fn_x({type,'sets'},{input_scales,output_scale},as,bs);



K = @(as,bs) fcov({type,'sets'},{input_scales,output_scale},as,bs)