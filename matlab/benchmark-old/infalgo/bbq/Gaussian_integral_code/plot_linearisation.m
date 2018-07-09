function draw_from_linearisation

plotdir = '~/Docs/bayes-quadrature-for-gaussian-integrals-paper/figures/';



colours = cbrewer('qual','Paired', 12);
colours = colours(2:2:end, :);

% NB: results seem fairly insensitive to selection of a*(sigh). b
% has desired effect.

% define observed likelihood, lik, & locations of predictants
% ====================================

% multiply fixed likelihood function by this constant
%const = 10^(5*(rand-.5));

%lik = rand(6,1)/const;
lik = ([0.1;0.25;0.1;0.15;0.18;0.15;0.48;0.05]);%/const;
n = length(lik);
x = linspace(0,10,n)';

sub_xst = -5;
add_xst = 5;

n_st = 1000;
xst = linspace(min(x) + sub_xst, max(x) + add_xst, n_st)';

% define map f(tlik) = lik
% ====================================

mn = min(lik);
mx = max(lik);


% delta in [0, 1] sets how wide the error bars are for small likelihoods:
% we expect the smallest possible likelihood is mn * delta
delta = 0.8;

alpha = 0; % arbitrary, wlog
gamma = 1; % arbitrary, wlog


% Maximum likelihood approach to finding map

f_h = @(tlik, alpha, gamma) gamma * (tlik - alpha).^2 + delta * mn;
df_h = @(tlik, alpha, gamma) 2 * gamma * (tlik - alpha);
invf_h = @(lik, alpha, gamma) sqrt((lik-delta * mn) / gamma) + alpha;


f = @(tlik) f_h(tlik, alpha, gamma);
df = @(tlik) df_h(tlik, alpha, gamma);
invf = @(lik) invf_h(lik, alpha, gamma);

figure(3)
clf
n_tliks = 1000;

% maximum observed transformed likelihood
beta = invf(mx);


tliks = linspace(alpha, beta, n_tliks);
plot(tliks,f(tliks),'k');
hold on
%plot(tliks,df(tliks),'r');

xlabel 'transformed likelihood'
ylabel 'likelihood'
set(gca, 'TickDir', 'out')
set(gca, 'Box', 'off', 'FontSize', 10); 
set(gcf, 'color', 'white'); 
set(gca, 'YGrid', 'off');
set(gca, 'color', 'white'); 

% define gp over inv-transformed (e.g. log-) likelihoods
% ====================================

invf_lik = invf(lik);

% gp covariance hypers
w = 1;
h = std(invf_lik);
sigma = eps;
mu = min(invf_lik);

% define covariance function
fK = @(x, xd) h^2 .* exp(-0.5*(x - xd).^2 / w^2);
K = @(x, xd) bsxfun(fK, x, xd');

% Gram matrix
V = K(x, x) + eye(n)*sigma^2;

% GP posterior for tlik
m_tlik = mu + K(xst, x) * (V\(invf_lik-mu));
C_tlik = K(xst, xst) - K(xst, x) * (V\K(x, xst));
sd_tlik = sqrt(diag(C_tlik));

figure(1)
clf
subplot(2, 1, 1)
params.legend_location = 0;
params.y_label = sprintf('transformed\n likelihood');
params.x_label = 0;
gp_plot(xst, m_tlik, sd_tlik, x, invf(lik), [], [], params);
axis tight;


% define linearisation, lik = a * tlik + c
% ====================================


% exact linearisation
lin_slope = df(m_tlik);
lin_const = f(m_tlik) - lin_slope .* m_tlik;

% unnecessary under the quadratic transform
% % approximate linearisation
% best_tlik = mu;
% a = df(best_tlik) + (m_tlik - best_tlik) .* ddf(best_tlik);
% c = f(m_tlik) - a .* m_tlik;

% gp over likelihood
% ====================================

m_lik = diag(lin_slope) * m_tlik + lin_const;
C_lik =  diag(lin_slope) * C_tlik * diag(lin_slope);
sd_lik = sqrt(diag(C_lik));

figure(1)
subplot(2, 1, 2)
params.legend_location = 0;
params.y_label = 'likelihood';
params.x_label = '$x$';
gp_plot(xst, m_lik, sd_lik, x, lik, [], [], params);
axis tight;
%plot(xst, a,'g')


% plot linearisation
% ====================================

min_plotted_tlik = m_tlik(1) - 2*sd_tlik(1);
ind = 0;
for i = round((-sub_xst + [-1, 5, 7.9])/(range(xst)) * n_st)
    ind = ind + 1;
    colour = colours(ind, :);
    
    figure(3)
    
    % plot linearisations
    x_vals = linspace(m_tlik(i) -  0.5 *sd_tlik(i), m_tlik(i) + 0.5*sd_tlik(i), 100);
    y_vals = lin_slope(i) * x_vals + lin_const(i);
    
    plot(x_vals, y_vals, 'Color', colour, 'LineWidth', 2)
    plot(m_tlik(i), m_lik(i), '.','MarkerSize', 20, 'Color', colour);
    
    
    % plot Gaussians in transformed likelihood
    x_vals = linspace(m_tlik(i) - 3 * sd_tlik(i), ...
        m_tlik(i) + 3 * sd_tlik(i), 100);
    y_vals = normpdf(x_vals, m_tlik(i), sd_tlik(i)) ...
        * .04 * (mx - mn) * (beta - alpha);
  
    plot(x_vals, y_vals, 'Color', colour);
    
    % plot approximate Gaussians in likelihood
    
    y_vals = linspace(m_lik(i) - 3 * sd_lik(i), m_lik(i) +  3 *sd_lik(i), 100);
    x_vals = min_plotted_tlik + normpdf(y_vals, m_lik(i), sd_lik(i)) ...
        * .04 * (mx - mn) * (beta - alpha);
    
    plot(x_vals, y_vals, 'Color', colour);
    
    % plot exact distributions in likelihood
    
    y_vals = linspace(1.2 * delta * mn, max(f(tliks)), 10000);
    ty_vals = invf(y_vals);
    x_vals = min_plotted_tlik + normpdf(ty_vals, m_tlik(i), sd_tlik(i)) ...
        ./ abs( df_h(ty_vals, alpha, gamma) )...
          * 0.04 * (mx - mn) * (beta - alpha);
    
    plot(x_vals, y_vals, '--', 'Color', colour);
    
    
    % indicate positions of these Gaussians in GPs over likelihood and
    % transformed likelihood
    
    figure(1)
    subplot(2, 1, 1)
    
    plot([xst(i), xst(i)], [m_tlik(i) - 2 * sd_tlik(i), m_tlik(i) + 2 * sd_tlik(i)], ...
        'LineWidth', 2, 'Color', colour);

    figure(1)
    subplot(2, 1, 2)
    
    plot([xst(i), xst(i)], [m_lik(i) - 2 * sd_lik(i), m_lik(i) + 2 * sd_lik(i)], ...
        'LineWidth', 2, 'Color', colour);
    
end

figure(3)
xlim([min_plotted_tlik, max(m_tlik)]);
ylim([0, mx]);

set(0, 'defaulttextinterpreter', 'none')
matlabfrag([plotdir,'lik_v_tlik'])

figure(1)
matlabfrag([plotdir,'gps_lik_tlik'])

close all

end

