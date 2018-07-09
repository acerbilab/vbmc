function [K] = WS_cov_fn(hps_struct,hp)

%Nhps = length(hp);

%if isnumeric(hps_struct)
%     % inputs are as Nsensors,hp,flag
%     
%     Nsensors = hps_struct;
%     Ncorrhps=Nsensors*(Nsensors-1)*0.5;
%     
%     corr_input_ind = 3;
%     indiv_input_scale_ind = 5;
%     delay_inds = 5+(1:Nsensors);
%     indiv_output_scale_inds = 5 + Nsensors + (1:Nsensors);
%     corr_inds = 5 + 2*Nsensors + (1:Ncorrhps);

%else   
    corr_input_ind = hps_struct.logCorrInputScale;
    indiv_input_scale_ind = hps_struct.logIndivInputScale;
    indiv_output_scale_inds = hps_struct.logIndivOutputScales;
    delay_inds = hps_struct.Delays;
    corr_inds = hps_struct.CorrelationNos;
    
    num_sensors = length(delay_inds);
%end

corr_input_scale = exp(hp(corr_input_ind));
indiv_input_scale = exp(hp(indiv_input_scale_ind));
indiv_output_scales = exp(hp(indiv_output_scale_inds));
if size(indiv_output_scales,2)>size(indiv_output_scales,1)
    indiv_output_scales=indiv_output_scales';
end
indiv_output_scales=allcombs({indiv_output_scales,nan}); % i have to do this to correct matlab's vector-indexing-vectors problem
delays = hp(delay_inds);
if size(delays,2)>size(delays,1)
    delays=delays';
end
delays=allcombs({delays,nan}); % i have to do this to correct matlab's vector-indexing-vectors problem
corrvec = hp(corr_inds);
log_corr_output_scales = corrvec(1:num_sensors);

Indep = length(corrvec) == num_sensors;

if Indep
    corrCov=diag(exp(2*log_corr_output_scales)); %Indep   
else
    corrCov=tri2(corrvec); %Dep
    %bareCov=tri2([zeros(1,Nsensors),corrAngles]);
    %justScales=corrCov./bareCov;
end


K=@(as,bs) matrify(@(al,at,bl,bt)...
    fcov('matern',{corr_input_scale,1,1/2},at-delays(al),bt-delays(bl))...
    .*corrCov((bl-1)*num_sensors+al)...
    +fcov('matern',{indiv_input_scale,indiv_output_scales(al),3/2},at,bt)...
    .*(al==bl),...
    as,bs);

