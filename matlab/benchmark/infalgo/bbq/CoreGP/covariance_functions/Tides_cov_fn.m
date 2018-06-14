function [K,out2] = Tides_cov_fn(hps_struct,hp,flag)



if isnumeric(hps_struct)
    % inputs are as num_sensors,hp,flag
    
    num_sensors = hps_struct;
    Ncorrhps=num_sensors*(num_sensors-1)*0.5;
    
    period_ind = 3;
    dist_output_scale_ind = 4;
    dist_input_scale_ind = 5;
    delay_inds = 5+(1:num_sensors);
    corr_inds = 5 + num_sensors + (1:num_sensors+Ncorrhps);

else   
    
    period_ind = hps_struct.logPeriod;
    dist_input_scale_ind = hps_struct.logDistInputScale;
    dist_output_scale_ind = hps_struct.logDistOutputScale;
    delay_inds = hps_struct.Delays;

    corr_inds = hps_struct.CorrelationNos;
    
end

num_sensors = length(delay_inds);

period = exp(hp(period_ind));
dist_output_scale = exp(hp(dist_output_scale_ind));
dist_input_scale = exp(hp(dist_input_scale_ind));
delays = hp(delay_inds);
if size(delays,2)>size(delays,1)
    delays=delays';
end
delays=allcombs({delays,nan}); % i have to do this to correct matlab's vector-indexing-vectors problem
corrvec = hp(corr_inds);
log_dist_output_scales = corrvec(1:num_sensors);

Indep = length(corrvec) == num_sensors;

if Indep
    corrMat=diag(exp(2*log_dist_output_scales)); %Indep   
else
    corrMat=tri2(corrvec); %Dep
    %bareCov=tri2([zeros(1,num_sensors),corrAngles]);
    %justScales=corrMat./bareCov;
end

K=@(as,bs) matrify(@(al,at,bl,bt)...
    (fcov('matern',{dist_input_scale,dist_output_scale,5/2},at-delays(al),bt-delays(bl))...
    +fcov({'matern','periodic'},{period,1,5/2},at-delays(al),bt-delays(bl)))...    
    .*corrMat((bl-1)*num_sensors+al),...
    as,bs);





% i used a slightly modified covariance, incidentally, it's a sum of a
% sqdexp and a periodic component, but the periodic component is a
% product of a sqdexp and the usual sqdexp of a sinl.  so the second
% term is periodic with a decay, which fixed lots of (mostly
% conditioning) problems.  plus it just makes more sense.  input scale,
% output scale, period, amplitude, decay, noise set with ML and some
% fmincons.
                
% (covfnper(TimeCov,at-Params{2}(al),bt-Params{2}(bl),Params{1},1,5/2)+...
%     covfn(DistTimeCov,at-Params{2}(al),bt-Params{2}(bl),Params{4},Params{5},5/2)).*Params{3}((bl-1)*GD+al);



% Kfn= @(al,at,bl,bt,Params)...
%     (covfnper(TimeCov,at-Params{2}(al),bt-Params{2}(bl),Params{1},1,5/2))...
%     .*Params{3}((bl-1)*GD+al)...
%     +covfn(DistTimeCov,at-Params{2}(al),bt-Params{2}(bl),Params{4},Params{5},5/2)...
%     .*Params{8}((bl-1)*GD+al)...
%     +(al==bl).*covfn(DistTimeCov,at-Params{2}(al),bt-Params{2}(bl),Params
%     {6},Params{7},3/2)


if nargin<3
    flag='no deriv';
end

if strcmpi(flag,'deriv hyperparams')        
    
    num_hps = length(hp);
    % this only correctly supplies the derivatives wrt the delays.
    out2=@(as,bs) DphiK(as,bs,dist_input_scale,dist_output_scale,delays,period,num_sensors,delay_inds,num_hps,corrMat);
end


function DphiKcell = DphiK(as,bs,dist_input_scale,dist_output_scale,delays,period,num_sensors,delay_inds,num_hps,corrMat)

num_rows = size(as,1);
num_cols = size(bs,1);

DphiKcell = mat2cell2d(zeros(num_hps*num_rows,num_cols),num_rows*ones(num_hps,1),num_cols);

als = as(:,1);
bls = bs(:,1);

first_term = matrify(@(al,at,bl,bt) gcov('matern',{dist_input_scale,dist_output_scale,5/2},at-delays(al),bt-delays(bl)),as,bs);
second_term = matrify(@(al,at,bl,bt) gcov({'matern','periodic'},{period,1,5/2},at-delays(al),bt-delays(bl)),as,bs);

first_term = -first_term{1};
second_term = -bsxfun(@(x,y) corrMat((x-1)*num_sensors+y),als,bls').*second_term{1};

for delay = 1:delay_inds
    DphiKcell{delay} = bsxfun(@(x,y) or(x==delay,y==delay),als,bls').*(first_term+second_term);
end

% function DphiKcell=DphiK(Xs1,Xs2,Indep,num_sensors,Nangles,Nhps,...
%     period_ind,...
%     amplitude_ind,...
%     dist_input_scale_ind,...
%     delay_inds,...
%     dist_output_scale_inds,...
%     corr_inds,...
%     period,...
%     amplitude,...
%     dist_input_scale,...
%     delays,...
%     log_dist_output_scales,...
%     corrAngles,...
%     corrvec)
% 
% % This does not return the correct derivativesx wrt any hps other than the
% % correlation angles
% 
% num_rows = size(Xs1,1);
% num_cols = size(Xs2,1);
% 
% No_effect=zeros(num_rows,num_cols);
% 
% K_mat = matrify(@(al,at,bl,bt)...
%     fcov('matern',{dist_input_scale,1,5/2},at-delays(al),bt-delays(bl)),...
%     Xs1,Xs2);
% 
% % Kmat_time=matrify(@(al,at,bl,bt)...
% %     cov(type,{T,1},at,bt),Xs1,Xs2);
% % Kmat_label=matrify(@(al,at,bl,bt)...
% %     SensorCov((bl-1)*num_sensors+al),Xs1,Xs2);
% % 
% % deriv_logInputScales=...
% %     matrify(@(al,at,bl,bt) gTcov(type,{T,1},at,bt),Xs1,Xs2);
% % deriv_logInputScales={deriv_logInputScales{1}.*Kmat_label};
% %                     
% %[Labels1,Labels2]=meshgrid2d(Xs1(:,1),Xs2(:,1));
% Labels1 = Xs1(:,1);
% Labels2 = Xs2(:,1);
% 
% % 
% % sensor=kron2d((1:num_sensors)',ones(size(Labels1)));
% % 
% % test1=repmat(Labels1,num_sensors,1)==sensor;
% % test2=repmat(Labels2,num_sensors,1)==sensor;
% % 
% % % Equal to 2 if both test1 and test2 are true, equal to 1 if one is true,
% % % zero otherwise
% % deriv_mat=test1+test2;
% % deriv_mat=deriv_mat.*repmat(Kmat_time.*Kmat_label,num_sensors,1);
% % deriv_logOutputScales=mat2cell2d(deriv_mat,num_rows*ones(1,num_sensors),num_cols);
% 
% DphiKcell = mat2cell2d(zeros(Nhps*num_rows,num_cols),num_rows*ones(Nhps,1),num_cols);
% 
% if ~Indep
%     Th=nan(1,6);
%     Th(1:Nangles)=corrAngles;
%     DTh=nan(4,4,Nangles);
% 
%     if Nangles>0
%     DTh(:,:,1)=[0 -sin(Th(1)) 0 0;  
%     -sin(Th(1)) 0 -cos(Th(2))*sin(Th(1))+cos(Th(1))*cos(Th(3))*sin(Th(2)) -cos(Th(4))*sin(Th(1))+cos(Th(1))*cos(Th(5))*sin(Th(4));
%     0 -cos(Th(2))*sin(Th(1))+cos(Th(1))*cos(Th(3))*sin(Th(2)) 0 0;
%     0 -cos(Th(4))*sin(Th(1))+cos(Th(1))*cos(Th(5))*sin(Th(4)) 0 0];
%     if Nangles>1
%     DTh(:,:,2)=[0 0 -sin(Th(2)) 0;  
%     0 0 cos(Th(2))*cos(Th(3))*sin(Th(1))-cos(Th(1))*sin(Th(2)) 0
%     -sin(Th(2)) cos(Th(2))*cos(Th(3))*sin(Th(1))-cos(Th(1))*sin(Th(2)) 0 (-cos(Th(4))*sin(Th(2))+cos(Th(2))*sin(Th(4)))*(cos(Th(3))*cos(Th(5))+cos(Th(6))*sin(Th(3))*sin(Th(5)))
%     0 0 (-cos(Th(4))*sin(Th(2))+cos(Th(2))*sin(Th(4)))*(cos(Th(3))*cos(Th(5))+cos(Th(6))*sin(Th(3))*sin(Th(5))) 0];
%     if Nangles>2
%     DTh(:,:,3)=[0 0 0 0;
%     0 0 -sin(Th(1))*sin(Th(2))*sin(Th(3)) 0;
%     0 -sin(Th(1))*sin(Th(2))*sin(Th(3)) 0 (cos(Th(2))*cos(Th(4))+sin(Th(2))*sin(Th(4)))*(-cos(Th(5))*sin(Th(3))+cos(Th(3))*cos(Th(6))*sin(Th(5)));
%     0 0 (cos(Th(2))*cos(Th(4))+sin(Th(2))*sin(Th(4)))*(-cos(Th(5))*sin(Th(3))+cos(Th(3))*cos(Th(6))*sin(Th(5))) 0];
%     if Nangles>3
%     DTh(:,:,4)=[0 0 0 -sin(Th(4));
%     0 0 0 cos(Th(4))*cos(Th(5))*sin(Th(1))-cos(Th(1))*sin(Th(4));
%     0 0 0 (cos(Th(4))*sin(Th(2))-cos(Th(2))*sin(Th(4)))*(cos(Th(3))*cos(Th(5))+cos(Th(6))*sin(Th(3))*sin(Th(5)));
%     -sin(Th(4)) cos(Th(4))*cos(Th(5))*sin(Th(1))-cos(Th(1))*sin(Th(4)) (cos(Th(4))*sin(Th(2))-cos(Th(2))*sin(Th(4)))*(cos(Th(3))*cos(Th(5))+cos(Th(6))*sin(Th(3))*sin(Th(5))) 0];
%     if Nangles>4
%     DTh(:,:,5)=[0 0 0 0;
%     0 0 0 -sin(Th(1))*sin(Th(4))*sin(Th(5));
%     0 0 0 (cos(Th(2))*cos(Th(4))+sin(Th(2))*sin(Th(4)))*(cos(Th(5))*cos(Th(6))*sin(Th(3))-cos(Th(3))*sin(Th(5)));
%     0 -sin(Th(1))*sin(Th(4))*sin(Th(5)) (cos(Th(2))*cos(Th(4))+sin(Th(2))*sin(Th(4)))*(cos(Th(5))*cos(Th(6))*sin(Th(3))-cos(Th(3))*sin(Th(5))) 0];
%     if Nangles>5
%     DTh(:,:,6)=[0 0 0 0;
%     0 0 0 0;
%     0 0 0 -sin(Th(3))*(cos(Th(2))*cos(Th(4))+sin(Th(2))*sin(Th(4)))*sin(Th(5))*sin(Th(6));
%     0 0 -sin(Th(3))*(cos(Th(2))*cos(Th(4))+sin(Th(2))*sin(Th(4)))*sin(Th(5))*sin(Th(6)) 0];
%     end
%     end
%     end
%     end
%     end
%     end
% 
%     deriv_mat=DTh(Labels1,Labels2,1:Nangles).*repmat(K_mat,[1,1,Nangles]);
%     deriv_corrAngles=mat2cell(deriv_mat,num_rows,num_cols,ones(Nangles,1));
%     deriv_corrAngles = reshape(deriv_corrAngles,Nangles,1);
%     
%     deriv_logOutputScales = repmat(No_effect,[1,1,num_sensors]);
%     deriv_logOutputScales = mat2cell(deriv_logOutputScales,num_rows,num_cols,ones(num_sensors,1));
%     deriv_logOutputScales = reshape(deriv_logOutputScales,num_sensors,1);
% 
%     DphiKcell(corr_inds) = ...
%                 {deriv_logOutputScales,...
%                 deriv_corrAngles};       % corrAngles         
% end
%                 