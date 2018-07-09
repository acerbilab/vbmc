function Kmat=Kwderivrot(Xs1,Xs2,varargin)
% Here we allow observations of the derivative along any direction, as
% specified by the vector comprising the last half of each X entry.
covargs=varargin;

Kmat=nan(size(Xs1,1),size(Xs2,1));

NDims=size(Xs1,2)/2;
if NDims~=floor(NDims)
    error('Derivative must be specified along a vector of dimension equal to that of the input space.');
end

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
    Kmat(obsL,obsR)=matrify(@(varargin) fcov(covargs{:},varargin{:}),...
        locsL(obsL,:),locsR(obsR,:));
end
if some_obsL && some_gobsR
    arrR=-permute(repmat(dirnsR(gobsR,:),1,1,NobsL),[3,1,2]);
    Kmat(obsL,gobsR)=...
        sumup(arrR,... %NB: negative sign due to derivative operation from right
        matrify(@(varargin) gcov(covargs{:},varargin{:}),...
        locsL(obsL,:),locsR(gobsR,:)));
end
if some_gobsL && some_obsR
    arrL=permute(repmat(dirnsL(gobsL,:),1,1,NobsR),[1,3,2]);
    Kmat(gobsL,obsR)=...
        sumup(arrL,...
        matrify(@(varargin) gcov(covargs{:},varargin{:}),...
        locsL(gobsL,:),locsR(obsR,:)));
end
if some_gobsL && some_gobsR
    arrL=permute(repmat(dirnsL(gobsL,:),1,1,NgobsR),[1,3,2]);
    arrR=-permute(repmat(dirnsR(gobsR,:),1,1,NgobsL),[3,1,2]);
    Kmat(gobsL,gobsR)=...
        sumup(repmat(arrR,1,1,NDims)...
             .*reshape(repmat(arrL,1,NDims),NgobsL,NgobsR,NDims^2),...
        matrify(@(varargin) Hcov(covargs{:},varargin{:}),...
        locsL(gobsL,:),locsR(gobsR,:)));
end




function S=sumup(arr,ce)
stack=arr.*cat(3,ce{:});
S=sum(stack,3);