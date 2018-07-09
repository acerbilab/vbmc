function A = meanVoightProfile(hyp, x, i)

% report number of hyperparameters
if (nargin < 2)
  A = '0';
  return;
end

eta = 0.5; %hyp;

gamma = (1 / 6);
sigma = gamma * sqrt(2 * log(2)) / 2;

cauchy_pdf = @(x) ((gamma ./ ((x - 0.5).^2 + gamma^2)) / pi);

if (nargin == 2)
  A =      eta  * cauchy_pdf(x) + ...
      (1 - eta) * normpdf(x, 0.5, sigma);
  A = A - (eta * cauchy_pdf(0) + (1 - eta) * normpdf(0, 0.5, sigma));
  A = -A / (eta * cauchy_pdf(0.5) + (1 - eta) * normpdf(0.5, 0.5, sigma));;
else
  A = zeros(size(x)); %cauchy_pdf(x) - normpdf(x, 0, sigma);
end
