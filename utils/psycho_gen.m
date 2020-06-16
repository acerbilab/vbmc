function R=psycho_gen(theta,S)
%PSYCHO_GEN Generate responses for psychometric function model.
%  R=PSYCHO_GEN(THETA,S) generates responses in a simple orientation
%  discrimination task, where S is a vector of stimulus orientations (in
%  deg) for each trial, and THETA is a model parameter vector, with 
%  THETA(1) as eta=log(sigma), the log of the sensory noise; THETA(2) the 
%  bias term; THETA(3) is the lapse rate. The returned vector of responses
%  per trial reports 1 for "rightwards" and -1 for "leftwards".
%
%  See Section 5.2 of the manuscript for more details on the model.
%
%  Note that this model is very simple and used only for didactic purposes;
%  one should use the analytical log-likelihood whenever available.

% Luigi Acerbi, 2020

sigma = exp(theta(1));
bias = theta(2);
lapse = theta(3);

%% Noisy measurement

% Ass Gaussian noise to true orientations S to simulate noisy measurements
X = S + sigma*randn(size(S));

%% Decision rule

% The response is 1 for "rightwards" if the internal measurement is larger
% than the BIAS term; -1 for "leftwards" otherwise
R = zeros(size(S));
R(X >= bias) = 1;
R(X < bias) = -1;

%% Lapses

% Choose trials in which subject lapses; response there is given at chance
lapse_idx = rand(size(S)) < lapse;

% Random responses (equal probability of 1 or -1)
lapse_val = randi(2,[sum(lapse_idx),1])*2-3;
R(lapse_idx) = lapse_val;

end