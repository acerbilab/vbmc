function Kout=DTKwderivrot(Xs1,Xs2,varargin)
% Derivative of Kwderivrot wrt input scales
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
    
    Kcell=matrify(@(varargin) gTcov(covargs{:},varargin{:}),...
        locsL(obsL,:),locsR(obsR,:));
    Kmat(obsL,obsR,:)=cat(3,Kcell{:});
end
if some_obsL && some_gobsR
    arrR=-permute(repmat(dirnsR(gobsR,:),1,1,NobsL),[3,1,2]);
    
    Kcell=matrify(@(varargin) gTgcov(covargs{:},varargin{:}),...
        locsL(obsL,:),locsR(gobsR,:));
    Kmat(obsL,gobsR,:)=reshape(...
        sum(repmat(arrR,1,NDims).*cell2mat(reshape(Kcell,1,NDims,NDims)),3),...
        NobsL,NgobsR,NDims);
    
end
if some_gobsL && some_obsR
    arrL=permute(repmat(dirnsL(gobsL,:),1,1,NobsR),[1,3,2]);

    Kcell=matrify(@(varargin) gTgcov(covargs{:},varargin{:}),...
        locsL(gobsL,:),locsR(obsR,:));
    Kmat(gobsL,obsR,:)=reshape(...
        sum(repmat(arrL,1,NDims).*cell2mat(reshape(Kcell,1,NDims,NDims)),3),...
        NgobsL,NobsR,NDims);
    
end
if some_gobsL && some_gobsR
    arrL=permute(repmat(dirnsL(gobsL,:),1,1,NgobsR),[1,3,2]);
    arrR=-permute(repmat(dirnsR(gobsR,:),1,1,NgobsL),[3,1,2]);
    arr=repmat(repmat(arrR,1,1,NDims)...
             .*reshape(repmat(arrL,1,NDims),NgobsL,NgobsR,NDims^2),...
             1,NDims);

    Kcell=matrify(@(varargin) gTHcov(covargs{:},varargin{:}),...
        locsL(gobsL,:),locsR(gobsR,:));
    Kmat(gobsL,gobsR,:)=reshape(...
        sum(arr.*cell2mat(reshape(Kcell,1,NDims,NDims^2)),3),...
        NgobsL,NgobsR,NDims);
end

Kout=squeeze(mat2cell3d(Kmat,NRows,NCols,ones(1,NDims)));
