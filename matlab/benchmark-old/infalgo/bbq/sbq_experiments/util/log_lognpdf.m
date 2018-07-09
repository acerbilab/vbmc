function logp = log_lognpdf(log_x,mu,sigma_squared)

if nargin<1
    error(message('stats:lognpdf:TooFewInputs'));
end
if nargin < 2
    mu = 0;
end
if nargin < 3
    sigma_squared = 1;
end

% Return NaN for out of range parameters.
sigma_squared(sigma_squared <= 0) = NaN;

try
    logp = -0.5 * ((log_x - mu).^2)./sigma_squared ...
         - log_x - log(sqrt(2*pi)) - 0.5*log(sigma_squared);
catch
    error(message('stats:lognpdf:InputSizeMismatch'));
end
