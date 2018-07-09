% Just a simple demo to double check equations for matching the moments of the
% log of a Gaussian distributed variable by a Gaussian.

mu_0 = 10;
v0 = 2;

xrange = 0.1:0.01:15;

% Original Gaussian
plot( xrange, mvnpdf(xrange', mu_0, v0), 'b'); hold on;

% Log of that Gaussian (an exp-normal dist).
plot( xrange, mvnpdf(exp(xrange)', mu_0, v0), 'g'); hold on;

% Moment-matching equations.
v_log = log( v0 / mu_0^2 + 1 )
u_log  = log(mu_0) - 0.5.*log(v0/(mu_0^2) + 1)

% Moment-matched Gaussian.
plot( xrange, mvnpdf(xrange', u_log, v_log), 'r'); hold on;

% exp of the moment-matched Gaussian (a log-normal dist.)
plot( xrange, mvnpdf(log(xrange)', u_log, v_log), 'k'); hold on;
