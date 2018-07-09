
N=1000;

q = -8.5;
p_pulse = 0.09;
R = nan; %0.1*pi;
K = nan;
decay = 0.9994;

[t, theta, direction] = simulate_ring_mf(N, R, K, p_pulse, decay, q);

save simulated_mf_data;
