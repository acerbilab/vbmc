function [K,out2] = sensor2term_cov_fn(Nsensors,type1,type2,hp,flag)

% This function is not even remotely finished

% TimeCov='matern';
% DistTimeCov='matern';

% Kfn= @(al,at,bl,bt,Params)...
%     (covfnper(TimeCov,at-Params{2}(al),bt-Params{2}(bl),Params{1},1,5/2))...
%     .*Params{3}((bl-1)*GD+al)...
%     +covfn(DistTimeCov,at-Params{2}(al),bt-Params{2}(bl),Params{4},Params{5},5/2)...
%     .*Params{8}((bl-1)*GD+al)...
%     +(al==bl).*covfn(DistTimeCov,at-Params{2}(al),bt-Params{2}(bl),Params{6},Params{7},3/2)

% for SampleInd=1:NSamples % Index over our set of Samples of parameters
%     
%         TimeScale=exp(Samples(SampleInd,1)); % TimeScale for this sample       
%         SensorDelays=[[0;Samples(SampleInd,2:GD)'],nan(GD,1);nan(1,2)]; % Sensor delays for this sample
%         % I had to make this a matrix rather than a vector in order to
%         % combat the annoying A(rowvector)=A(column vector) if A is itself
%         % just a vector
%         if Indep
%             SensorCov=diag(exp(2*Samples(SampleInd,GD+1:2*GD)));
%             BareCov=eye(GD);
%         else
%             SensorCov=tri2(Samples(SampleInd,GD+1:GD+rSize));
%             BareCov=tri2([zeros(1,GD),Samples(SampleInd,2*GD+1:GD+rSize)]);
%         end
%         TimeScale2=exp(Samples(SampleInd,rSize+GD+1)); % TimeScale for disturbances
%         LS2=exp(Samples(SampleInd,rSize+GD+2)); % LengthScale for disturbance signal
%         TimeScale3=exp(Samples(SampleInd,rSize+GD+3)); % TimeScale for disturbances
%         LS3=exp(Samples(SampleInd,rSize+GD+4)); % LengthScale for disturbance signal
%         
%         ObsSD=exp(Samples(SampleInd,end)); % Observation noise for this sample
%         
%         Params{SampleInd}={TimeScale,SensorDelays,SensorCov,TimeScale2,LS2,TimeScale3,LS3,BareCov,ObsSD};
% end


T1=exp(hp(3));
 
%SensorDelays=[[0;Samples(SampleInd,2:Nsensors)'],nan(Nsensors,1);nan(1,2)]; % Sensor delays for this sample
% I had to make this a matrix rather than a vector in order to
% combat the annoying A(rowvector)=A(column vector) if A is itself
% just a vector

NcorrCovhps=Nsensors*(Nsensors+1)*0.5;
vec=hp(3+(1:NcorrCovhps));
Nangles=length(vec)-Nsensors;
Indep=Nangles==0;
corrAngles=vec(Nsensors+1:end);

if Indep
    corrCov=diag(exp(2*vec)); %Indep   
    bareCov=eye(Nsensors);
else
    corrCov=tri2(vec); %Dep
    bareCov=tri2([zeros(GD,1),vec(GD+1,end)]);
end
%posn:posn+0.5*Nsensors*(Nsensors+1)-1));


T2=exp(hp(5+NcorrCovhps));
IndivCov=diag(exp(2*hp(6+NcorrCovhps+(1:Nsensors))));

K=@(as,bs) matrify(@(al,at,bl,bt)...
    fcov(type1,{T1,1},at,bt)...
    .*corrCov((bl-1)*Nsensors+al)...
    +fcov(type2,{T2,1},at,bt)...
    .*IndivCov((bl-1)*Nsensors+al),as,bs);


if nargin<4
    flag='deriv inputs';
end

if strcmpi(flag,'deriv hyperparams')               
    out2=@(Xs1,Xs2) DphiK(Xs1,Xs2,Indep,Nsensors,Nangles,type,T,corrCov,corrAngles);
end

function DphiKcell=DphiK(Xs1,Xs2,Indep,Nsensors,Nangles,type,T,corrCov,corrAngles)

No_effect=zeros(size(Xs1,1),size(Xs2,1));

Kmat_time1=matrify(@(al,at,bl,bt)...
    fcov(type1,{T1,1},at,bt),Xs1,Xs2);
Kmat_label1=matrify(@(al,at,bl,bt)...
    corrCov((bl-1)*Nsensors+al),Xs1,Xs2);

Kmat_time1=matrify(@(al,at,bl,bt)...
    fcov(type1,{T1,1},at,bt),Xs1,Xs2);
Kmat_label1=matrify(@(al,at,bl,bt)...
    corrCov((bl-1)*Nsensors+al),Xs1,Xs2);

deriv_logInputScales1=...
    matrify(@(al,at,bl,bt) gTcov(type1,{T1,1},at,bt),Xs1,Xs2);
deriv_logInputScales1={deriv_logInputScales1{1}.*matrify(@(al,at,bl,bt) corrCov((bl-1)*Nsensors+al),Xs1,Xs2)};

deriv_logInputScales2=...
    matrify(@(al,at,bl,bt) gTcov(type2,{T2,1},at,bt),Xs1,Xs2);
deriv_logInputScales2={deriv_logInputScales2{1}.*matrify(@(al,at,bl,bt) IndivCov((bl-1)*Nsensors+al),Xs1,Xs2)};
                    

                    
[Labels1,Labels2]=meshgrid2d(Xs1(:,1),Xs2(:,1));
[Nrows,Ncols]=size(Labels1);

sensor=kron2d((1:Nsensors)',ones(size(Labels1)));

test1=repmat(Labels1,Nsensors,1)==sensor;
test2=repmat(Labels2,Nsensors,1)==sensor;

% Equal to 2 if both test1 and test2 are true, equal to 1 if one is true,
% zero otherwise
deriv_mat=test1+test2;
deriv_mat=deriv_mat.*repmat(Kmat_time.*Kmat_label,Nsensors,1);
deriv_logOutputScales=mat2cell2d(deriv_mat,Nrows*ones(1,Nsensors),Ncols);


if Indep
    DphiKcell=[{No_effect;...              % mean
                No_effect};...             % logNoiseSD           
                deriv_logInputScales;...  % logInputScales
                deriv_logOutputScales];  % logOutputScales
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
    0 0 cos(Th(2))*cos(Th(3))*Sin(Th(1))-cos(Th(1))*sin(Th(2)) 0
    -sin(Th(2)) Cos(Th(2))*Cos(Th(3))*Sin(Th(1))-cos(Th(1))*sin(Th(2)) 0 (-cos(Th(4))*sin(Th(2))+cos(Th(2))*Sin(Th(4)))*(cos(Th(3))*cos(Th(5))+cos(Th(6))*sin(Th(3))*sin(Th(5)))
    0 0 (-cos(Th(4))*sin(Th(2))+cos(Th(2))*Sin(Th(4)))*(cos(Th(3))*cos(Th(5))+cos(Th(6))*sin(Th(3))*sin(Th(5))) 0];
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

    deriv_mat=DTh(Labels1,Labels2,1:Nangles).*repmat(Kmat_label,1,1,Nangles);
    deriv_corrAngles=mat2cell3d(deriv_mat,Nrows,Ncols,ones(Nangles,1));

    DphiKcell=[{No_effect;...              % mean
                No_effect};...             % logNoiseSD           
                deriv_logInputScales;...  % logInputScales
                deriv_logOutputScales;      % logOutputScales
                deriv_corrAngles];       % corrAngles         
end
                
