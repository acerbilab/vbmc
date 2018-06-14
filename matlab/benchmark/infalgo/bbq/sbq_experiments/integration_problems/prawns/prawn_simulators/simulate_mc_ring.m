function fake_experiments = simulate_mc_ring(N, nump, params, modelidx)


    
    if nargin <3
        modelidx = 1;
    end



switch modelidx
    case 0
        sim_fn= @(x) simulate_ring_null(nump, x(1), x(2), x(3:4), x(5), x(6));
    case 1
        sim_fn= @(x) simulate_ring_mf(nump, x(1), x(2), x(3:4), x(5), x(6));
    case 2
        sim_fn= @(x) simulate_ring_topo(nump, x(1), x(2), x(3:4), x(5), x(6));
    case 3
        sim_fn= @(x) simulate_ring_R(nump, x(1), x(2), x(3:4), x(5), x(6));
    case 4
        sim_fn= @(x) simulate_ring_R2ways(nump, x(1), x(2), x(3:4), x(5), x(6));
    case 5
        sim_fn= @(x) simulate_ring_R_ahead(nump, x(1), x(2), x(3:4), x(5), x(6));
    case 6
        sim_fn= @(x) simulate_ring_R_ahead2ways(nump, x(1), x(2), x(3:4), x(5), x(6));       
    case 7
        sim_fn= @(x) simulate_ring_memory(nump, x(1), x(2), x(3:4), x(5), x(6));
    case 8
        sim_fn= @(x) simulate_ring_memory2ways(nump, x(1), x(2), x(3:4), x(5), x(6));     
    case 9
        sim_fn= @(x) simulate_ring_memory_ahead(nump, x(1), x(2), x(3:4), x(5), x(6));
    case 10
        sim_fn= @(x) simulate_ring_memory_ahead2ways(nump, x(1), x(2), x(3:4), x(5), x(6));   
end


parfor i =1:N
    [~, theta, directions] = sim_fn(params);
    for j = 1:nump
        fake_experiments(i).theta{j} = theta(j, :)';
        fake_experiments(i).sequence{j} = (directions(j, :)'+3)/2;
    end
end





