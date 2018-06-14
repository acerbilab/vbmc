function [t, theta, direction] = simulate_ring_R2ways(N, R, K, p_pulse, decay, q)

colr = {'or', 'ob', 'ok'};
% q = -8.5;
% p_pulse = 0.09;
% decay = 0.9994;
% R = 0.1*pi;

%params that work: q=-8.5, pp = 0.5, d = 0.9882, R = 0.1*pi, dt = 1/15

p = zeros(N, N);

interaction_M = zeros(N, N);

if nargin < 7
    plot_flag =0;
end

dt = 1/7.5; % time step
w_m = normrnd(3*pi*(dt*15)/180, 2*(dt*15)*pi/180, N,1); %average Angular speed of each prawn0.75
w_s = normrnd(0*pi/180, 0*pi/180, N, 1); %std 0.3


T_max = 360;

theta = zeros(N, ceil(T_max / dt));
state = zeros(N, ceil(T_max / dt));

theta(:, 1) = 2 * pi * rand(N, 1); %initial positions
D = zeros(ceil(T_max / dt), 1);
direction = (rand(N, 1) > 0.5) * 2 - 1; %Sets directions as 1 or -1 with prob 0.5.

t = 0; %init time counter
count = 0;




%while(abs(sum(direction))/N < 0.9 && t < T_max)
while(t < T_max)
    count = count+1;
    t = t + dt;
    w = normrnd(direction.*w_m, w_s); %Set theta jump for each prawn
   
    
    
    %Determine possible interactions, change directions with Prob p and
    %record which prawns have already interacted.
    rand_ord = randperm(N);
    for ri = 1:N
        i = rand_ord(ri); %update prawns in random order to avoid artifacts
        
        theta(i, count+1) = mod(theta(i, count) + w(i), 2*pi);
        [gap, choice_idx] = min([abs(theta(:, count)-theta(i, count)), 2*pi-(abs(theta(:, count)-theta(i, count)))], [], 2);
        position = zeros(N, 1);
        position(choice_idx == 1) = direction(i)*sign(theta(choice_idx==1, count)-theta(i, count));
        position(choice_idx == 2) = -direction(i)*sign(theta(choice_idx==2, count)-theta(i, count));

       
      sd = 0;
        for j = 1:N
            if i ~= j && gap(j) < R
               
                if direction(i) == -direction(j)
                    sd = sd + p_pulse(1);
                else 
                    
                
                end
            
            end
        end
       
        
        if rand < 1/(1+exp(-sd-q))
            direction(i) = -direction(i);
            
        end
        
        
    end
 
    
    state(:, count) = direction;
    
end

theta = theta(:, 1:count);
direction = state;
state = (state(:, 2:count)+3)/2;







