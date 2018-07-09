load sixinputs

tic

for j = 1:10
    for i = 0:10
    
       [lp{i+1, j}, smp{i+1, j}] = logP_mc_ring_memory(theta, directions, 10000, i, 1);
    end
    j
    toc
    
    save prawn_mc_results_corr lp smp
end

close all;clear; make_ML_fig;

parfor i = 0:10
    pr = mean(squeeze(meanp(i+1, :, :)));
    
    fake_exps(i+1).three_prawn_data = simulate_mc_ring(1000,3,pr, i);
    fake_exps(i+1).six_prawn_data = simulate_mc_ring(1000,6,pr, i);
    fake_exps(i+1).twelve_prawn_data = simulate_mc_ring(1000,12,pr, i);
    
    
    
   
end
toc
     save prawn_mc_results_corr lp smp fake_exps
    