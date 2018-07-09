function Kmat=Kwderiv(Xs1,Xs2,varargin)
covargs=varargin;

Kmat=nan(size(Xs1,1),size(Xs2,1));

obsL=Xs1(:,end)==0;
obsR=Xs2(:,end)==0;

some_obsL=~all(~obsL);
some_obsR=~all(~obsR);

% the final column of Xs1 and Xs2 contains an integer labelling the whether
% an observation is a derivative and which variable it is the partial
% derivative wrt.
derivsL=setdiff(unique(Xs1(:,end)),0);
derivsR=setdiff(unique(Xs2(:,end)),0);

gobsL=false(size(Xs1,1),max(derivsL));
for derivL=derivsL
    gobsL(:,derivL)=Xs1(:,end)==derivL;
end
gobsR=false(size(Xs2,1),max(derivsR));
for derivR=derivsR
    gobsR(:,derivR)=Xs2(:,end)==derivR;
end

if some_obsL && some_obsR
    Kmat(obsL,obsR)=matrify(@(varargin) cov(covargs{:},varargin{:}),...
        Xs1(obsL,1:end-1),Xs2(obsR,1:end-1));
end
if some_obsL 
    for derivR=derivsR
        Kmat(obsL,gobsR(:,derivR))=matrify(@(varargin) -g(derivR,covargs,varargin{:}),...
            Xs1(obsL,1:end-1),Xs2(gobsR(:,derivR),1:end-1));
    end
end
if some_obsR
    for derivL=derivsL
        Kmat(gobsL(:,derivL),obsR)=matrify(@(varargin) g(derivL,covargs,varargin{:}),...
            Xs1(gobsL(:,derivL),1:end-1),Xs2(obsR,1:end-1));
    end
end
for derivL=derivsL
    for derivR=derivsR
        Kmat(gobsL(:,derivL),gobsR(:,derivR))=matrify(@(varargin) -H(derivL,derivR,covargs,varargin{:}),...
            Xs1(gobsL(:,derivL),1:end-1),Xs2(gobsR(:,derivR),1:end-1));
    end
end

% NB: if two or more derivative observations are at the same locations, the
% functions below are a bit wasteful - we compute the relevant matrix
% multiple times.

function out=g(deriv,covargs,varargin)

out=gcov(covargs{:},varargin{:});
out=out{deriv};

function out=H(derivL,derivR,covargs,varargin)

out=Hcov(covargs{:},varargin{:});
out=out{derivL,derivR};


