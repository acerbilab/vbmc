function [ p, e ] = qsclatmvnv( m, r, a, cn, b )
%
%  [ P E ] = QSCLATMVNV( M, R, A, CN, B )
%    uses a randomized quasi-random rule with m points to estimate an
%    MVN probability for positive semi-definite covariance matrix r,
%    with constraints a < cn*x < b. If r is nxn and cn is kxn, then
%    a and b must be column k-vectors.
%   Probability p is output with error estimate e.
%    Example use:
%     >> r = [ 4 3 2 1; 3 5 -1 1; 2 -1 4 2; 1 1 2 5 ];
%     >> a = [ -inf 1 -5 ]'; b = [ 3 inf 4 ]';
%     >> cn = [ 1 2 3 -2; 2 4 1 2; -2 3 4 1 ];
%     >> [ p e ] = qsclatmvnv( 5000, r, a, cn, b ); disp([ p e ])
%
%  This function uses an algorithm given in the paper by Alan Genz:
%   "Numerical Computation of Multivariate Normal Probabilities", in
%     J. of Computational and Graphical Stat., 1(1992), 141-149.
%  The primary references for the numerical integration are 
%   "Randomization of Number Theoretic Methods for Multiple Integration"
%    R. Cranley and T.N.L. Patterson, SIAM J Numer Anal, 13(1976), pp. 904-14.
%  and  
%   "Fast Component-by-Component Construction, a Reprise for Different 
%    Kernels", D. Nuyens and R. Cools. In H. Niederreiter and D. Talay,
%    editors, Monte-Carlo and Quasi-Monte Carlo Methods 2004, 
%    Springer-Verlag, 2006, 371-385.
%
%   Alan Genz is the author of this function and following Matlab functions.
%          Alan Genz, WSU Math, PO Box 643113, Pullman, WA 99164-3113
%          Email : alangenz@wsu.edu
%
% Initialization
% 
[ as ch bs clg n ] = chlsrt( r, a, cn, b ); 
ns = 10; [ q pr ] = fstrnk( m/ns, n-1 ); q = q/pr;
ai = min(          max( as(1), -9 )  , 9 ); c = phi(ai);
bi = min( max( ai, max( bs(1), -9 ) ), 9 ); d = phi(bi);
ci = c; dci = d - ci; y = zeros(n-1,pr); p = 0; e = 0; 
%
% Randomization loop for ns samples
%
for S = 1 : ns, c = ci; dc = dci; vp = dc; li = 2; lf = 1;
  %
  % Compute randomized MVN integrand with tent periodization
  %
  for i = 2 : n, lf = lf + clg(i); 
    y(i-1,:) = phinv( c + abs( 2*mod(q(i-1)*[1:pr]+rand,1) - 1 ).*dc ); 
    if lf < li, c = 0; dc = 1;
    else, ai = -9; bi = 9;
      for j = li : lf, s = ch(j,1:i-1)*y(1:i-1,:); 
	ai = max( as(j) - s, ai ); bi = min( bs(j) - s, bi ); 
      end, ai = min( ai, 9 ); bi = max( ai, max( bi, -9 ) );
      c = phi(ai); dc = phi(bi) - c; vp = vp.*dc; 
    end, li = li + clg(i);
  end, d = ( mean(vp) - p )/S; p = p + d; 
  if abs(d) > 0, e = abs(d)*sqrt( 1 + ( e/d )^2*( S - 2 )/S );
  else, if S > 1, e = e*sqrt( ( S - 2 )/S ); end
  end
end, e = 3*e; % error estimate is 3 x standard error with ns samples.
%
% end qsclatmvnv
%
function [ ap, ch, bp, clg, np ] = chlsrt( r, a, cn, b )
%
%  Computes permuted lower Cholesky factor ch for covariance r which 
%   may be singular, combined with contraints a < cn*x < b, to
%   form revised lower triangular constraint set ap < ch*x < bp; 
%   clg contains information about structure of ch: clg(1) rows for 
%   ch with 1 nonzero, ..., clg(np) rows with np nonzeros.
%
ep = 1e-10; % singularity tolerance;
%
[n n] = size(r); c = r; [m n] = size(cn); ch = cn; dc = sqrt(max(diag(c),0));
for i = 1 : n, d = dc(i);
  if d > 0, c(:,i) = c(:,i)/d; c(i,:) = c(i,:)/d; ch(:,i) = ch(:,i)*d; end
end, ap = a; bp = b; np = 0; 
%
% determine (with pivoting) Cholesky factor for r 
%  and form revised constraint matrix ch
%
for i = 1 : n, clg(i) = 0; epi = ep*i^2; j = i; 
  for l = i+1 : n, if c(l,l) > c(j,j), j = l; end, end
  if j > i, t = c(i,i); c(i,i) = c(j,j); c(j,j) = t;
    t = c(i,1:i-1); c(i,1:i-1) = c(j,1:i-1); c(j,1:i-1) = t;
    t = c(i+1:j-1,i); c(i+1:j-1,i) = c(j,i+1:j-1)'; c(j,i+1:j-1) = t';
    t = c(j+1:n,i); c(j+1:n,i) = c(j+1:n,j); c(j+1:n,j) = t;
    t = ch(:,i); ch(:,i) = ch(:,j); ch(:,j) = t;
  end
  if c(i,i) < epi, break, end, cvd = sqrt( c(i,i) ); c(i,i) = cvd;
  for l = i+1 : n
    c(l,i) = c(l,i)/cvd; c(l,i+1:l) = c(l,i+1:l) - c(l,i)*c(i+1:l,i)';
  end, ch(:,i) = ch(:,i:n)*c(i:n,i); np = np + 1;
end, y = zeros(n,1); sqtp = sqrt(2*pi); 
%
% use right reflectors to reduce ch to lower triangular
%
for i = 1 : min( np-1, m ), epi = ep*i^2; vm = 1 + epi; lm = i;
  %
  % permute rows so that smallest variance variables are first.
  %
  for l = i : m, v = ch(l,1:np); s = v(1:i-1)*y(1:i-1); 
    ss = max( sqrt( sum( v(i:np).^2 ) ), epi ); 
    al = ( ap(l) - s )/ss; bl = ( bp(l) - s )/ss; 
    dna = 0; dsa = 0; dnb = 0; dsb = 1;
    if al > -9, dna = exp(-al*al/2)/sqtp; dsa = phi(al); end
    if bl <  9, dnb = exp(-bl*bl/2)/sqtp; dsb = phi(bl); end, p = dsb - dsa;
    if p > epi, mn = dna - dnb; vr = al*dna - bl*dnb; 
      if      al <= -9, mn = -dnb; vr = -bl*dnb;
      elseif  bl >=  9, mn =  dna; vr =  al*dna; 
      end, mn = mn/p; vr = 1 + vr/p - mn^2;
    else, vr = 0; mn = ( al + bl )/2;
      if al < -9, mn = bl; elseif bl > 9, mn = al; end
    end, if vr <= vm, lm = l; vm = vr; y(i) = mn; end
  end, v = ch(lm,1:np);
  if lm > i, ch(lm,1:np) = ch(i,1:np); ch(i,1:np) = v;
    tl = ap(i); ap(i) = ap(lm); ap(lm) = tl;
    tl = bp(i); bp(i) = bp(lm); bp(lm) = tl;
  end, ch(i,i+1:np) = 0; ss = sum( v(i+1:np).^2 );
  if ss > epi, ss = sqrt( ss + v(i)^2 ); if v(i) < 0, ss = -ss; end
    ch(i,i) = -ss; v(i) = v(i) + ss; vt = v(i:np)'/( ss*v(i) );
    ch(i+1:m,i:np) = ch(i+1:m,i:np) - ch(i+1:m,i:np)*vt*v(i:np); 
  end
end
%
% scale and sort constraints
%
for i = 1 : m, v = ch(i,1:np); clm(i) = min(i,np); 
  jm = 1; for j = 1 : clm(i), if abs(v(j)) > ep*j*j, jm = j; end, end 
  if jm < np, v(jm+1:np) = 0; end, clg(jm) = clg(jm) + 1; 
  at = ap(i); bt = bp(i); j = i;
  for l = i-1 : -1 : 1
    if jm >= clm(l), break, end
    ch(l+1,1:np) = ch(l,1:np); j = l;
    ap(l+1) = ap(l); bp(l+1) = bp(l); clm(l+1) = clm(l);
  end, clm(j) = jm; vjm = v(jm); ch(j,1:np) = v/vjm; 
  ap(j) = at/vjm; bp(j) = bt/vjm;
  if vjm < 0, tl = ap(j); ap(j) = bp(j); bp(j) = tl; end
end, j = 0; for i = 1 : np, if clg(i) > 0, j = i; end, end, np = j;
%
% combine constraints for first variable
%
if clg(1) > 1 
  ap(1) = max( ap(1:clg(1)) ); bp(1) = max( ap(1), min( bp(1:clg(1)) ) ); 
  ap(2:m-clg(1)+1) = ap(clg(1)+1:m); bp(2:m-clg(1)+1) = bp(clg(1)+1:m);
  ch(2:m-clg(1)+1,:) = ch(clg(1)+1:m,:); clg(1) = 1;
end
%
% end chlsrt
%
function p = phi(z)
%
%  Standard statistical normal distribution cdf
%
p = erfc( -z/sqrt(2) )/2;
return
%
% end phi
%
function z = phinv(w)
%
%  Standard statistical inverse normal distribution
%
z = -sqrt(2)*erfcinv( 2*w );
return
%
% end phinv
%
function [ z, n ] = fstrnk( ni, sm, om, gm, bt )
% 
% Reference: 
%   "Fast Component-by-Component Construction, a Reprise for Different 
%     Kernels", D. Nuyens and R. Cools. In H. Niederreiter and D. Talay,
%     editors, Monte-Carlo and Quasi-Monte Carlo Methods 2004, 
%     Springer-Verlag, 2006, 371-385.
% Modifications to original by A. Genz, 05/07
% Typical Use:  
%  om = inline('x.^2-x+1/6'); n = 99991; s = 100; gam = 0.9.^[1:s];
%  z = fastrank( n, s, om, gam, 1 + gam/3 ); disp([z'; e])
%
n = fix(ni); if ~isprime(n), pt = primes(n); n = pt(length(pt)); end
if nargin < 3, om = inline('x.^2-x+1/6'); 
  bt = ones(1,sm); gm = [ 1 (4/5).^[0:sm-2] ]; 
end, q = 1; w = 1; z = [1:sm]'; m = ( n - 1 )/2; g = prmrot(n); 
perm = [1:m]; for j = 1 : m-1, perm(j+1) = mod( g*perm(j), n ); end
perm = min( n - perm, perm ); c = om(perm/n); fc = fft(c); 
for s = 2 : sm, q = q.*( bt(s-1) + gm(s-1)*c([w:-1:1 m:-1:w+1]) );
  [ es w ] = min( real( ifft( fc.*fft(q) ) ) ); z(s) = perm(w); 
end
%
% end fstrnk
%
function r = prmrot(pin)
%
% find primitive root for prime p, might fail for large primes (p > 32e7)
%
p = pin; if ~isprime(p), pt = primes(p); p = pt(length(pt)); end
pm = p - 1; fp = unique(factor(pm)); n = length(fp); r = 2; k = 1;
while k <= n; d = pm/fp(k); rd = r;
  for i = 2 : d, rd = mod( rd*r, p ); end % computes r^d mod p
  if rd == 1, r = r + 1; k = 0; end, k = k + 1;
end    
%
% prmrot
%
