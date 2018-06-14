function [ p, e, ef, efe ] = qsimvnef( m, r, a, b, f )
%
%  [ P E EF EFE ] = QSIMMVNEF( M, R, A, B )
%    uses a randomized quasi-random rule with m points to estimate an
%    MVN expectation for positive definite covariance matrix r,
%    with lower integration limits a and upper integration limits b,
%    and expectation function f.  
%   Probability MVN p is output with error estimate e, along with
%      expected value ef and error estimate efe.
%      Note: ef approx.= E[f] = I[f]/p = I[f]/I[1], where I[.]
%         denotes the truncated MVN integral.
%    Example usage:
%     r = [4 3 2 1;3 5 -1 1;2 -1 4 2;1 1 2 5];
%     a = -inf*[1 1 1 1 ]'; b = [ 1 2 3 4 ]';
%     f = inline('x(1)^2*x(2)*x(3)*x(4)','x');
%     [ p e ef efe ] = qsimvnef( 50000, r, a, b, f ); disp([ p e ef efe ])
%   Note: if f is defined in an m-file f.m, then use
%     [p e ef efe] = qsimvnef( 50000, r, a, b, 'f' ); disp([ p e ef efe ]) 
%    which is usually faster than the inline f.
%
%   This function uses an algorithm given in the paper
%      "Numerical Computation of Multivariate Normal Probabilities", in
%      J. of Computational and Graphical Stat., 1(1992), pp. 141-149, by
%          Alan Genz, WSU Math, PO Box 643113, Pullman, WA 99164-3113
%          Email : AlanGenz@wsu.edu
%  The primary references for the numerical integration are 
%   "On a Number-Theoretical Integration Method"
%   H. Niederreiter, Aequationes Mathematicae, 8(1972), pp. 304-11, and
%   "Randomization of Number Theoretic Methods for Multiple Integration"
%    R. Cranley and T.N.L. Patterson, SIAM J Numer Anal, 13(1976), pp. 904-14.
%
%   Alan Genz is the author of this function and following Matlab functions.
%
% Initialization
%
[n, n] = size(r); ch = r'; as = a; bs = b;
ct = ch(1,1); ai = as(1); bi = bs(1); 
c = ( 1 + sign(ai) )/2; if abs(ai) < 9*ct, c = phi(ai/ct); end
d = ( 1 + sign(bi) )/2; if abs(bi) < 9*ct, d = phi(bi/ct); end   
ci = c; dci = d - ci; p = 0; e = 0; ef = 0; efe = 0;
ns = 8; nv = max( [ m/( 2*ns ) 1 ] ); 
q = 2.^( [1:n]'/(n+1)) ; % Niederreiter point set generators
%
% Randomization loop for ns samples
%
for i = 1 : ns
  vi = 0;%zeros(n+1,1); 
  xr = rand( n, 1 ); 
  %
  % Loop for 2*nv quasirandom points
  %
  for  j = 1 : nv
    x = abs( 2*mod( j*q + xr, 1 ) - 1 ); % periodizing transformation
    vp =   mvndnse( n, ch, ci, dci,   x, as, bs, f ); 
    vp = ( mvndnse( n, ch, ci, dci, 1-x, as, bs, f ) + vp )/2; 
    vi = vi + ( vp - vi )/j; 
  end   
  %
  d = ( vi(1) - p )/i; p = p + d; 
  if abs(d) > 0 
    e = abs(d)*sqrt( 1 + ( e/d )^2*( i - 2 )/i );
  else
    if i > 1, e = e*sqrt( ( i - 2 )/i ); end
  end
  df = ( vi(2:end) - ef )/i; ef = ef + df; efe = efe*( i - 2 )/i + df'*df; 
end
%
e = 3*e; % error estimate is 3 x standard error with ns samples.
ef = ef/p; efe = 3*sqrt(efe)/p; 
%
% end qsimvn
%
function  pe = mvndnse( n, ch, ci, dci, x, a, b, f )
%
%  Transformed integrand for computation of MVN expectations. 
%
y = zeros(n,1); s = 0; c = ci; dc = dci; p = dc; 
for i = 2 : n
  y(i-1) = phinv( c + x(i-1)*dc ); s = ch(i,1:i-1)*y(1:i-1); 
  ct = ch(i,i); ai = a(i) - s; bi = b(i) - s;
  c = ( 1 + sign(ai) )/2; if abs(ai) < 9*ct, c = phi(ai/ct); end
  d = ( 1 + sign(bi) )/2; if abs(bi) < 9*ct, d = phi(bi/ct); end 
  dc = d - c; p = p*dc; 
end 
y(n) = phinv( c + x(n)*dc ); pe = [ p; p*feval( f, ch*y ) ]; 
%
% end mvndnse
%
function p = phi(z)
%
%  Standard statistical normal distribution
%
p = erfc( -z/sqrt(2) )/2;
%
% end phi
%
function z = phinv(w)
%
%  Standard statistical inverse normal distribution
%
z = -sqrt(2)*erfcinv( 2*w );
%
% end phinv

