function [K,out2] = AT_cov_fn(Nsensors,hp,flag)

T1=exp(hp(3));
T2=exp(hp(4));
L2=exp(hp(5));
T3=exp(hp(6));
L3=exp(hp(7));

vec=hp(8:end);
Nangles=length(vec)-Nsensors;
Indep=Nangles==0;
corrAngles=vec(Nsensors+1:end);

if Indep
    corrCov=diag(exp(2*vec)); %Indep   
    bareCov=eye(Nsensors);
else
    corrCov=tri2(vec); %Dep
    bareCov=tri2([zeros(1,Nsensors),corrAngles]);
    justScales=corrCov./bareCov;
end
%posn:posn+0.5*Nsensors*(Nsensors+1)-1));

% No delays necessary
K=@(as,bs) matrify(@(al,at,bl,bt)...
    fcov({'matern','periodic'},{T1,1,5/2},at,bt)...
    .*corrCov((bl-1)*Nsensors+al)...
    +fcov('matern',{T2,L2,5/2},at,bt)...
    .*bareCov((bl-1)*Nsensors+al)...    
    +fcov('matern',{T3,L3,3/2},at,bt)...
    .*(al==bl),...
    as,bs);
                
% Kfn= @(al,at,bl,bt,Params)...
%     (covfnper(TimeCov,at-Params{2}(al),bt-Params{2}(bl),Params{1},1,5/2))...
%     .*Params{3}((bl-1)*GD+al)...
%     +covfn(DistTimeCov,at-Params{2}(al),bt-Params{2}(bl),Params{4},Params{5},5/2)...
%     .*Params{8}((bl-1)*GD+al)...
%     +(al==bl).*covfn(DistTimeCov,at-Params{2}(al),bt-Params{2}(bl),Params
%     {6},Params{7},3/2)


if nargin<3
    flag='deriv inputs';
end

if strcmpi(flag,'deriv hyperparams')               
    out2=@(Xs1,Xs2) DphiK(Xs1,Xs2,Indep,Nsensors,Nangles,T1,T2,L2,T3,L3,justScales,corrAngles);
end

function DphiKcell=DphiK(Xs1,Xs2,Indep,Nsensors,Nangles,T1,T2,L2,T3,L3,justScales,corrAngles)

% This does not return the correct derivativesx wrt any hps other than the
% correlation angles

No_effect=zeros(size(Xs1,1),size(Xs2,1));
[Nrows,Ncols]=size(No_effect);

K_mat = matrify(@(al,at,bl,bt)...
    fcov({'matern','periodic'},{T1,1,5/2},at,bt)...
    .*justScales((bl-1)*Nsensors+al)...
    +fcov('matern',{T2,L2,5/2},at,bt),...    
    Xs1,Xs2);

% Kmat_time=matrify(@(al,at,bl,bt)...
%     cov(type,{T,1},at,bt),Xs1,Xs2);
% Kmat_label=matrify(@(al,at,bl,bt)...
%     SensorCov((bl-1)*Nsensors+al),Xs1,Xs2);
% 
% deriv_logInputScales=...
%     matrify(@(al,at,bl,bt) gTcov(type,{T,1},at,bt),Xs1,Xs2);
% deriv_logInputScales={deriv_logInputScales{1}.*Kmat_label};
%                     
%[Labels1,Labels2]=meshgrid2d(Xs1(:,1),Xs2(:,1));
Labels1 = Xs1(:,1);
Labels2 = Xs2(:,1);

% 
% sensor=kron2d((1:Nsensors)',ones(size(Labels1)));
% 
% test1=repmat(Labels1,Nsensors,1)==sensor;
% test2=repmat(Labels2,Nsensors,1)==sensor;
% 
% % Equal to 2 if both test1 and test2 are true, equal to 1 if one is true,
% % zero otherwise
% deriv_mat=test1+test2;
% deriv_mat=deriv_mat.*repmat(Kmat_time.*Kmat_label,Nsensors,1);
% deriv_logOutputScales=mat2cell2d(deriv_mat,Nrows*ones(1,Nsensors),Ncols);


if Indep
    DphiKcell=[{No_effect;...              % mean
                No_effect;...             % logNoiseSD           
                No_effect;...  
                No_effect;...  
                No_effect;...
                No_effect;...
                No_effect}];  
else
    Th=nan(1,6);
    Th(1:Nangles)=corrAngles;
    DTh=nan(4,4,Nangles);

    if Nangles>0
    DTh(:,:,1)=[0 -sin(Th(1)) 0 0;  
    -sin(Th(1)) 0 -cos(Th(2))*sin(Th(1))+cos(Th(1))*cos(Th(3))*sin(Th(2)) -cos(Th(4))*sin(Th(1))+cos(Th(1))*cos(Th(5))*sin(Th(4));
    0 -cos(Th(2))*sin(Th(1))+cos(Th(1))*cos(Th(3))*sin(Th(2)) 0 0;
    0 -cos(Th(4))*sin(Th(1))+cos(Th(1))*cos(Th(5))*sin(Th(4)) 0 0];
    if Nangles>1
    DTh(:,:,2)=[0 0 -sin(Th(2)) 0;  
    0 0 cos(Th(2))*cos(Th(3))*sin(Th(1))-cos(Th(1))*sin(Th(2)) 0
    -sin(Th(2)) cos(Th(2))*cos(Th(3))*sin(Th(1))-cos(Th(1))*sin(Th(2)) 0 (-cos(Th(4))*sin(Th(2))+cos(Th(2))*sin(Th(4)))*(cos(Th(3))*cos(Th(5))+cos(Th(6))*sin(Th(3))*sin(Th(5)))
    0 0 (-cos(Th(4))*sin(Th(2))+cos(Th(2))*sin(Th(4)))*(cos(Th(3))*cos(Th(5))+cos(Th(6))*sin(Th(3))*sin(Th(5))) 0];
    if Nangles>2
    DTh(:,:,3)=[0 0 0 0;
    0 0 -sin(Th(1))*sin(Th(2))*sin(Th(3)) 0;
    0 -sin(Th(1))*sin(Th(2))*sin(Th(3)) 0 (cos(Th(2))*cos(Th(4))+sin(Th(2))*sin(Th(4)))*(-cos(Th(5))*sin(Th(3))+cos(Th(3))*cos(Th(6))*sin(Th(5)));
    0 0 (cos(Th(2))*cos(Th(4))+sin(Th(2))*sin(Th(4)))*(-cos(Th(5))*sin(Th(3))+cos(Th(3))*cos(Th(6))*sin(Th(5))) 0];
    if Nangles>3
    DTh(:,:,4)=[0 0 0 -sin(Th(4));
    0 0 0 cos(Th(4))*cos(Th(5))*sin(Th(1))-cos(Th(1))*sin(Th(4));
    0 0 0 (cos(Th(4))*sin(Th(2))-cos(Th(2))*sin(Th(4)))*(cos(Th(3))*cos(Th(5))+cos(Th(6))*sin(Th(3))*sin(Th(5)));
    -sin(Th(4)) cos(Th(4))*cos(Th(5))*sin(Th(1))-cos(Th(1))*sin(Th(4)) (cos(Th(4))*sin(Th(2))-cos(Th(2))*sin(Th(4)))*(cos(Th(3))*cos(Th(5))+cos(Th(6))*sin(Th(3))*sin(Th(5))) 0];
    if Nangles>4
    DTh(:,:,5)=[0 0 0 0;
    0 0 0 -sin(Th(1))*sin(Th(4))*sin(Th(5));
    0 0 0 (cos(Th(2))*cos(Th(4))+sin(Th(2))*sin(Th(4)))*(cos(Th(5))*cos(Th(6))*sin(Th(3))-cos(Th(3))*sin(Th(5)));
    0 -sin(Th(1))*sin(Th(4))*sin(Th(5)) (cos(Th(2))*cos(Th(4))+sin(Th(2))*sin(Th(4)))*(cos(Th(5))*cos(Th(6))*sin(Th(3))-cos(Th(3))*sin(Th(5))) 0];
    if Nangles>5
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

    deriv_mat=DTh(Labels1,Labels2,1:Nangles).*repmat(K_mat,[1,1,Nangles]);
    deriv_corrAngles=mat2cell(deriv_mat,Nrows,Ncols,ones(Nangles,1));
    deriv_corrAngles = reshape(deriv_corrAngles,Nangles,1);
    
    deriv_logOutputScales = repmat(No_effect,[1,1,Nsensors]);
    deriv_logOutputScales = mat2cell(deriv_logOutputScales,Nrows,Ncols,ones(Nsensors,1));
    deriv_logOutputScales = reshape(deriv_logOutputScales,Nsensors,1);

    DphiKcell=[{No_effect;...              % mean
                No_effect;...             % logNoiseSD           
                No_effect;...  
                No_effect;...  
                No_effect;...
                No_effect;...
                No_effect};
                deriv_logOutputScales;...
                deriv_corrAngles];       % corrAngles         
end
                