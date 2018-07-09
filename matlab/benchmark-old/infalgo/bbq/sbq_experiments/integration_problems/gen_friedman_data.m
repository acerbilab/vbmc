function [X, y] = gen_friedman_data
% An attempted reconstruction of the dataset used in the 'Bayesian Monte Carlo'
% paper by Rasmussen and Ghahramani.
%
% David Duvenaud
% February 2012
% =======================

noiseless_func = ...
    @(x) 10*sin(pi * x(1)*x(2)) + 20 * (x(3) - 0.5)^2 + 10 * x(4) + 5 * x(5);

num_samples = 100;

rng(0);  % Set the random seed.

for i = 1:num_samples
    X(i, 1:5) = rand(1, 5);   % Choose an X;
    y(i) = noiseless_func(X(i, :)) + randn;
end
    
y = y';

save( 'friedman_data.mat', 'X', 'y');
