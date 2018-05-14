% Compute expected uncertainty

x = sym('x','real');
mu = sym('mu','real');
sigma = sym('sigma','positive'); 
z = sym('z','real');

npdf = exp(-(x-mu)^2/2/sigma^2)/sqrt(2*pi*sigma^2);
% fbar = x*exp(x)*npdf;

fbar = int(x*exp(x-z)*npdf,[-Inf,Inf])
f2bar = int(x^2*exp(2*(x-z))*npdf,[-Inf,Inf])

simplify(f2bar - fbar^2)