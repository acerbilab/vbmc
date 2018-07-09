function Kout=DKwderivrot(Xs1,Xs2,varargin)
% Gradient of Kwderivrot wrt first input.
covargs=varargin;

NDims=size(Xs1,2)/2;
if NDims~=floor(NDims)
    error('Derivative must be specified along a vector of dimension equal to that of the input space.');
end

NRows=size(Xs1,1);
NCols=size(Xs2,1);
Kmat=nan(NRows,NCols,NDims);

locsL=Xs1(:,1:NDims);
dirnsL=Xs1(:,NDims+1:end);
locsR=Xs2(:,1:NDims);
dirnsR=Xs2(:,NDims+1:end);

obsL=all(dirnsL==0,2);
obsR=all(dirnsR==0,2);

NobsL=sum(obsL);
NobsR=sum(obsR);

some_obsL=~all(~obsL);
some_obsR=~all(~obsR);

gobsL=~obsL;
gobsR=~obsR;

NgobsL=sum(gobsL);
NgobsR=sum(gobsR);

some_gobsL=~all(~gobsL);
some_gobsR=~all(~gobsR);

dirnsL=dirnsL./repmat(sqrt(dirnsL.^2*ones(NDims,1)),1,NDims);
dirnsR=dirnsR./repmat(sqrt(dirnsR.^2*ones(NDims,1)),1,NDims);

% derivsL=find(~all(dirnsL==0,1));
% derivsR=find(~all(dirnsR==0,1));

if some_obsL && some_obsR
    Kcell=matrify(@(varargin) gcov(covargs{:},varargin{:}),...
        locsL(obsL,:),locsR(obsR,:));
    Kmat(obsL,obsR,:)=cat(3,Kcell{:});
end
if some_obsL && some_gobsR
    arrR=-permute(repmat(dirnsR(gobsR,:),1,1,NobsL),[3,1,2]); %NB: negative sign due to derivative operation from right

    Kcell=matrify(@(varargin) Hcov(covargs{:},varargin{:}),...
        locsL(obsL,:),locsR(gobsR,:));
    Kmat(obsL,gobsR,:)=reshape(...
        sum(repmat(arrR,1,NDims).*cell2mat(reshape(Kcell,1,NDims,NDims)),3),...
        NobsL,NgobsR,NDims);
end
Kout=squeeze(mat2cell3d(Kmat,NRows,NCols,ones(1,NDims)));
