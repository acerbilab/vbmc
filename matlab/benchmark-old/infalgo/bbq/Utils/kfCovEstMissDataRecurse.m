% os = kfCovEstMissDataRecurse(X,Y,p,q,r,MissDataCode,AR);
%   - os struct with fields [A,y,E,St,Sinv,Sdeflate,Sdiags]
%	-runs Kalman-Bucy filter over observations matrix X
%	  for 1-step prediction onto matrix Y (Y can = X)
%     assumes X,Y are normalised to zero mean, unit variance columns
%	-with model order p, state noise variance q and obs noise variance r:
%     for X,Y normalised default suggestion is q=r=0.01
%     if -ve inputs are given then these are fixed and do not adapt
%	-returns model parameter sequence A,
%	  sequence of predicted outcomes y_pred
%	  and prediction covariances: E is just cov(y-ypred) and St is posterior
%	  covariance of observations. Sinv = inv(St) at each time step
%     Sdeflate is diagonal variances with mutual info removed from other
%     observations
%	(c) Stephen Roberts, Oct 2007

% AR=1, q=0.1, r=0.1, p = AR order, MissDataCode=-Inf

function os = kfCovEstMissDataRecurse(X,Y,p,q,r,MissDataCode,AR)

pOrig = p;
SEG = pOrig; % segment size is model order
D = size(X,2); % the dimensionality of the input stream
p = (p*D+1)*D ; % include a *separate* non-AR state for each stream
SHIFT = 1; % move on one timestep at a time
AlphaR = 0.7; % iir estimator for prediction error [subjective variable]
AlphaQ = 0.9; % iir estimator for prediction error [subjective variable]

% now set up state & observation noise components
if (q<0)
    q = -q;
    qAdapt = 0;
else
    qAdapt = 1;
end;

if (r<0)
    r = -r;
    rAdapt = 0;
else
    rAdapt = 1;
end;

% set up variables ahead of time
N = round(size(X,1)/SHIFT - SEG/SHIFT);
a = zeros(p,1);
R = eye(D)*r; % initial obs noise covariance
Q = eye(p)*q; % state noise covariance for diffusion
F = eye(p); % plant model - no systematic trend dynamics for now
B = 0; % no exogenous inputs
P = eye(p); % initial state covariance
I = eye(p); % identity
A = zeros(N,p);
E = zeros(N,D*D);
St = zeros(N,D*D);
Pt = zeros(N,p*p);
Sdeflate = zeros(N,D);
Sdiags = zeros(N,D);
y_pred = zeros(N,D);
adaptRate = zeros(N,1);
yPredOld = zeros(1,D); % last iteration's output prediction
Hstatic = zeros(SEG,D); % set to zero to start with

Qtot = zeros(1,N);
Rtot = zeros(D,D,N);
RCor = zeros(1,N);
qq = q;
% start iterating through data
for t = 1+SHIFT:N
  n = (t-1)*SHIFT + SEG +1;
  x_input = X(n-1,:);
  missInp = find(x_input == MissDataCode);
  if (AR==1)
      x_input(missInp) = yPredOld(missInp);
  else
      x_input(missInp) = 0;
  end;
  if (pOrig>0)
      Hstatic = [Hstatic(2:SEG,:) ; x_input]; % vector of past samples - combine with state vector to give predictions
      H = [1;reshape(Hstatic,prod(size(Hstatic)),1)]';
  else
      H = 1;
  end;
  
  H = kron(H,eye(D));
  
  %predict steps
  a = F*a + B; % form updated prior on state, a[k|k-1]
  %a = min(a,1); a = max(a,-1);
  P0 = F*P*F'; % state variance with no state diffusion added
  P = F*P*F' + Q; % updated prior on state covariance P[k|k-1]
  
  %update steps
  missLabel = find(Y(n,:) == MissDataCode);
  y = Y(n,:);		% one step true value read in from data
  y_pred(n,:) = H*a; % the observation prediction
  yPredOld = y_pred(n,:);
  y(missLabel) = y_pred(n,missLabel); % if the true label does not exist
  
  err_pred = (y - y_pred(n,:))'; % the error between prediction and true value
  upd_err = err_pred;
  %err_pred(missLabel) = y(missLabel);
  S = H*P*H' + R; % posterior covariance over y[k|k-1]
  %S1 = H*P*H';
  S0 = H*P0*H' + R; % posterior covariance over y[k|k-1] with no state diffusion
  Si = inv(S); % inverse of above
  K = P*H' * Si; % Kalman gain vector
  a = a + K*upd_err; % state posterior, i.e. a[k|k]
  P = (I - K*H)*P; % posterior state covariance P[k|k]
  Spost = H*P*H' + R; % full posterior for y[k|k]
  SpostInv = inv(Spost);
  Swu = H*P*H'; % covariance due to intrinsic model uncertainty
  pred_std = sqrt(diag(S));
  err_pred(missLabel) = pred_std(missLabel);
  if (rAdapt == 1)
      %R = AlphaR*R + (1-AlphaR)*(err_pred*err_pred' - S1);
      R = AlphaR*R + (1-AlphaR)*(err_pred*err_pred' - Swu); % rolling estimate of observation noise covariance
      %R = AlphaR*R + (1-AlphaR)*(err_pred*err_pred'); % rolling estimate of observation noise covariance
  end;
  
  if (qAdapt == 1)
      q = max(0, mean(diag(err_pred*err_pred' - S0))/mean(diag(H*H')));
      qq = AlphaQ*qq + (1-AlphaQ)*q;
      Q = qq*I;
  end;
  
  adaptRate(n) = mean(diag(P)); % average adaption rate is mean of diagonal of state cov matrix P

  Qtot(n) = qq;
  Rtot(:,:,n) = R;
  %RCor(n) = R(1,2);
  % re-ordering for output
  %E(n,:) = max(0,reshape(err_pred*err_pred' - Swu,1,D*D));
  %E(n,:) = reshape(err_pred*err_pred',1,D*D);
  E(n,:) = reshape(R,1,D*D);
  St(n,:) = reshape(Spost,1,D*D);
  Sdiags(n,:) = diag(Spost);
  Sinv(n,:) = reshape(SpostInv,1,D*D);
  Sdeflate(n,:) = covDeflate(Spost);
  A(n,:) = a;
  Pt(n,:) = reshape(P,1,p*p);
  
  % print reassuring dots
  if (rem(t,100)==0)
      fprintf('.');
  end;
end;

%fprintf('\n');

% set up the output structure [os]
os.A = A;
os.Pt = Pt;

%mean
os.y = y_pred;

os.E = E;
os.St = St;

% variance:
os.Sdiags = Sdiags;

os.Sinv = Sinv;
os.Sdeflate = Sdeflate;
os.ar = adaptRate;
os.Q = Qtot;
os.R = Rtot;
%os.C = RCor;

% function to look at MI overlap - collapse of variance in each stream due
% to information in all others via Sxx-Sxy inv(Syy) Sxy^T
function Sdef = covDeflate(S);

Sdiag = diag(S);
D = length(Sdiag);

for d=1:D,
    Sxx = Sdiag(d);
    if (d==1),
        Sxy = S(1,2:D);
        Syy = S(2:D,2:D);
    elseif (d==D)
        Sxy = S(D,1:D-1);
        Syy = S(1:D-1,1:D-1);
    else
        Sxy = S(d,[1:d-1 , d+1:D]);
        Syy = S([1:d-1 , d+1:D] , [1:d-1 , d+1:D]);
    end;
    
    Sdef(d) = Sxx - Sxy*inv(Syy)*Sxy';
end;