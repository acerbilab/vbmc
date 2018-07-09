function [K,out2] = sensor_cov_fn(num_sensors,hps_struct,type,hp,flag) 
% [K,out2] = sensor_cov_fn(num_sensors,hps_struct,type,hp,flag)  
    
num_hps = length(hp);
input_scale_inds = hps_struct.logInputScales;
T=exp(hp(input_scale_inds));

if isfield(hps_struct,'CorrelationNos')
    corr_inds = hps_struct.CorrelationNos;
    
    output_scale_inds = corr_inds(1:num_sensors);
    corr_angle_inds = corr_inds(num_sensors+1:end);
    
    num_angles = length(corr_angle_inds);
    
    corrvec = hp(corr_inds);
    log_corr_output_scales = hp(output_scale_inds);
    corr_angles = hp(corr_angle_inds);

    Indep = num_angles == 0;

    if Indep
        SensorCov=diag(exp(2*log_corr_output_scales)); %Indep   
    else
        SensorCov=tri2(corrvec); %Dep
        %bareCov=tri2([zeros(1,num_sensors),corr_angles]);
        %justScales=corrCov./bareCov;
    end
else
    % only one sensor
    output_scale_inds = hps_struct.logOutputScale;
    corr_angle_inds = [];
    corr_angles =[];
    num_angles = 0;
    
    Indep = true;
    
    SensorCov = exp(hp(output_scale_inds))^2;
end

    
if isfield(hps_struct,'Delays')
    delay_inds = hps_struct.Delays;
    delays = hp(delay_inds);
    if size(delays,2)>size(delays,1)
        delays=delays';
    end
else
    delays = zeros(num_sensors,1);
end
delays=allcombs({delays,nan}); % i have to do this to correct matlab's vector-indexing-vectors problem

K=@(as,bs) matrify(@(al,at,bl,bt)...
    fcov(type,{T,1},at-delays(al),bt-delays(bl))...
    .*SensorCov((bl-1)*num_sensors+al),as,bs);


if nargin<5
    flag='deriv inputs';
end

if strcmpi(flag,'deriv hyperparams')               
    out2=@(Xs1,Xs2) DphiK(Xs1,Xs2,Indep,num_sensors,num_angles,type,T,SensorCov,corr_angles,num_hps,input_scale_inds,output_scale_inds,corr_angle_inds);
end

function DK=DphiK(Xs1,Xs2,Indep,num_sensors,num_angles,type,T,SensorCov,corr_angles,num_hps,input_scale_inds,output_scale_inds,corr_angle_inds)


Kmat_time=matrify(@(al,at,bl,bt)...
    fcov(type,{T,1},at,bt),Xs1,Xs2);
Kmat_label=matrify(@(al,at,bl,bt)...
    SensorCov((bl-1)*num_sensors+al),Xs1,Xs2);

deriv_logInputScales=...
    matrify(@(al,at,bl,bt) gTcov(type,{T,1},at,bt),Xs1,Xs2);
deriv_logInputScales={deriv_logInputScales{1}.*Kmat_label};
                    
[Labels1,Labels2]=meshgrid2d(Xs1(:,1),Xs2(:,1));
[Nrows,Ncols]=size(Labels1);

sensor=kron2d((1:num_sensors)',ones(size(Labels1)));

test1=repmat(Labels1,num_sensors,1)==sensor;
test2=repmat(Labels2,num_sensors,1)==sensor;

% Equal to 2 if both test1 and test2 are true, equal to 1 if one is true,
% zero otherwise
deriv_mat=test1+test2;
deriv_mat=deriv_mat.*repmat(Kmat_time.*Kmat_label,num_sensors,1);
deriv_logOutputScales=mat2cell2d(deriv_mat,Nrows*ones(1,num_sensors),Ncols);

L1 = size(Xs1,1);
L2 = size(Xs2,1);

DK = mat2cell2d(zeros(num_hps*L1,L2),L1*ones(num_hps,1),L2);


if Indep
    DK(input_scale_inds) = deriv_logInputScales;
    DK(output_scale_inds) = deriv_logOutputScales;
else
    Th=nan(1,6);
    Th(1:num_angles)=corr_angles;
    DTh=nan(4,4,num_angles);

    if num_angles>0
    DTh(:,:,1)=[0 -sin(Th(1)) 0 0;  
    -sin(Th(1)) 0 -cos(Th(2))*sin(Th(1))+cos(Th(1))*cos(Th(3))*sin(Th(2)) -cos(Th(4))*sin(Th(1))+cos(Th(1))*cos(Th(5))*sin(Th(4));
    0 -cos(Th(2))*sin(Th(1))+cos(Th(1))*cos(Th(3))*sin(Th(2)) 0 0;
    0 -cos(Th(4))*sin(Th(1))+cos(Th(1))*cos(Th(5))*sin(Th(4)) 0 0];
    if num_angles>1
    DTh(:,:,2)=[0 0 -sin(Th(2)) 0;  
    0 0 cos(Th(2))*cos(Th(3))*Sin(Th(1))-cos(Th(1))*sin(Th(2)) 0
    -sin(Th(2)) Cos(Th(2))*Cos(Th(3))*Sin(Th(1))-cos(Th(1))*sin(Th(2)) 0 (-cos(Th(4))*sin(Th(2))+cos(Th(2))*Sin(Th(4)))*(cos(Th(3))*cos(Th(5))+cos(Th(6))*sin(Th(3))*sin(Th(5)))
    0 0 (-cos(Th(4))*sin(Th(2))+cos(Th(2))*Sin(Th(4)))*(cos(Th(3))*cos(Th(5))+cos(Th(6))*sin(Th(3))*sin(Th(5))) 0];
    if num_angles>2
    DTh(:,:,3)=[0 0 0 0;
    0 0 -sin(Th(1))*sin(Th(2))*sin(Th(3)) 0;
    0 -sin(Th(1))*sin(Th(2))*sin(Th(3)) 0 (cos(Th(2))*cos(Th(4))+sin(Th(2))*sin(Th(4)))*(-cos(Th(5))*sin(Th(3))+cos(Th(3))*cos(Th(6))*sin(Th(5)));
    0 0 (cos(Th(2))*cos(Th(4))+sin(Th(2))*sin(Th(4)))*(-cos(Th(5))*sin(Th(3))+cos(Th(3))*cos(Th(6))*sin(Th(5))) 0];
    if num_angles>3
    DTh(:,:,4)=[0 0 0 -sin(Th(4));
    0 0 0 cos(Th(4))*cos(Th(5))*sin(Th(1))-cos(Th(1))*sin(Th(4));
    0 0 0 (cos(Th(4))*sin(Th(2))-cos(Th(2))*sin(Th(4)))*(cos(Th(3))*cos(Th(5))+cos(Th(6))*sin(Th(3))*sin(Th(5)));
    -sin(Th(4)) cos(Th(4))*cos(Th(5))*sin(Th(1))-cos(Th(1))*sin(Th(4)) (cos(Th(4))*sin(Th(2))-cos(Th(2))*sin(Th(4)))*(cos(Th(3))*cos(Th(5))+cos(Th(6))*sin(Th(3))*sin(Th(5))) 0];
    if num_angles>4
    DTh(:,:,5)=[0 0 0 0;
    0 0 0 -sin(Th(1))*sin(Th(4))*sin(Th(5));
    0 0 0 (cos(Th(2))*cos(Th(4))+sin(Th(2))*sin(Th(4)))*(cos(Th(5))*cos(Th(6))*sin(Th(3))-cos(Th(3))*sin(Th(5)));
    0 -sin(Th(1))*sin(Th(4))*sin(Th(5)) (cos(Th(2))*cos(Th(4))+sin(Th(2))*sin(Th(4)))*(cos(Th(5))*cos(Th(6))*sin(Th(3))-cos(Th(3))*sin(Th(5))) 0];
    if num_angles>5
    DTh(:,:,6)=[0 0 0 0;
    0 0 0 0;
    0 0 0 -sin(Th(3))*(cos(Th(2))*cos(Th(4))+sin(Th(2))*sin(Th(4)))*sin(Th(5))*sin(Th(6));
    0 0 -sin(Th(3))*(cos(Th(2))*cos(Th(4))+sin(Th(2))*sin(Th(4)))*sin(Th(5))*sin(Th(6)) 0];
    end
    end
    end
    end
    end
    end

    deriv_mat=DTh(Labels1,Labels2,1:num_angles).*repmat(Kmat_time,1,1,num_angles);
    deriv_corr_angles=mat2cell3d(deriv_mat,Nrows,Ncols,ones(num_angles,1));

    DK(input_scale_inds) = deriv_logInputScales;
    DK(output_scale_inds) = deriv_logOutputScales;
    DK(corr_angle_inds) = deriv_corr_angles;
end
                
