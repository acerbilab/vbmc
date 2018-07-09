function logP = logP_ring_memory_2ways(theta, direction, R, K, p_pulse, decay, q)

if numel(p_pulse) == 1
    p_pulse(2) = -p_pulse(1);
end
if R < 0 || R > 2.1*pi || decay < 0 || decay > 1
    logP = -inf;
    return;
end

if iscell(theta)
    logP = 0;
    for i = 1:numel(theta)
        logP = logP + logP_ring_memory_2ways(theta{i}, direction{i}, R, K,p_pulse, decay, q);
    end
else

N = size(theta, 1);

p = zeros(N, N);

interaction_M = zeros(N, N);


logP = 0;






for count = 1:length(direction)-1
    
    for i = 1:N
        
        [gap, choice_idx] = min([abs(theta(:, count)-theta(i, count)), 2*pi-(abs(theta(:, count)-theta(i, count)))], [], 2);
        position = zeros(N, 1);
        position(choice_idx == 1) = direction(i, count)*sign(theta(choice_idx==1, count)-theta(i, count));
        position(choice_idx == 2) = -direction(i, count)*sign(theta(choice_idx==2, count)-theta(i, count));

        for j = 1:N
            if i == j
                %do nothing
            elseif interaction_M(i, j) == 1
                %turn off if no longer in interaction zone
                if gap(j) > R
                    interaction_M(i, j) = 0;
                end
                
            elseif interaction_M(i, j) == 0
                %turn on if in interaction zone and increase p
                if gap(j) < R
                    if direction(j, count)*direction(i, count) == 1
                        p(i,j) = p(i, j) + p_pulse(2);
                    else
                        p(i, j) = p(i, j) + p_pulse(1);
                    end
                    interaction_M(i,j) = 1;
                else
                    p(i, j) = p(i, j)*decay; %interaction over, strength decays
                end
                
            end
        end
        
        if direction(i, count+1) == direction(i, count)
            logP = logP + log(1 - 1/(1+exp(-sum(p(i, :))-q)));
        else
            logP = logP + log(1/(1+exp(-sum(p(i, :))-q)));
            p(i, :) = zeros(1, N);
        end
        
        
    end
    
end
       
    
    
end
    
  




