%%%%%%%%%%%%
% John P Cunningham
% 2012
%
% sample notes to run epmgp
%%%%%%%%%%%%

% sample sizes and seed
rng(0)
% dimensionality of normal
n = 100;
% number of polyhedral constraints
p = 73;

% make a Gaussian
m = zeros(n,1);
K = randn(n);
K = K*K' + eye(n);
% make a polyhedron
C = rand(n,p);
% normalize the directions
C = C./repmat(sqrt(sum(C.^2,1)),size(C,1),1);
% box boundaries
lB = -1*rand(p,1) - 2;
uB = rand(p,1) + 2;

% calculate with epmgp
[lZpolyWithEPMGP,mu,Sigma,extras] = epmgp(m,K,C,lB,uB);

lZpolyWithEPMGP

% genz method...
[lZpolyWithGenz, ~ , errGenz] = genzmgp(m,K,C,lB,uB,5000);

lZpolyWithGenz

% axis aligned...
C = eye(n);
lB = rand(n,1);
uB = rand(n,1) + 3;
[lZaxisWithEPMGP,mu,Sigma] = epmgp(m,K,C,lB,uB);

lZaxisWithEPMGP

% axis aligned code...
[lZaxisWithAxisEPMGP,mu,Sigma] = axisepmgp(m,K,lB,uB);

lZaxisWithAxisEPMGP

% sanity check
fprintf('epmgp and axisepmgp should produce the same answer for hyperrectangles.\n');
fprintf('ie this number should be close to zero: %g.\n',lZaxisWithEPMGP-lZaxisWithAxisEPMGP);

