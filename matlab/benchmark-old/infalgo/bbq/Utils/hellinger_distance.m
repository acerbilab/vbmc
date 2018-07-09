function distance = hellinger_distance(Kss, Ksa, Ksb, Va, Vb, Kab)
% distance = hellinger_distance(Kss, Ksa, Ksb, Ksa_invVa, Ksb_invVb, Kab)
% Kss =  K(predictants,predictants)
% Ksa = K(predictants,set_A)
% Ksb = K(predictants,set_B)
% Va = V(set_A,set_A)
% Vb = V(set_B,set_B)
% Ksa_invVa = K(predictants,set_A)*inv(V(set_A,set_A))
% Ksb_invVb = K(predictants,set_B)*inv(V(set_B,set_B))
% Kab = K(set_A,set_B)

opts.SYM = true;
opts.POSDEF = true;

Ksa_invVa = linsolve(Va, Ksa', opts)';
Ksb_invVb = linsolve(Vb, Ksb', opts)';

% Ksa_invVa = Ksa/Va;
% Ksb_invVb = Ksb/Vb;

pred_var_a = Kss - Ksa_invVa*Ksa';
pred_var_b = Kss - Ksb_invVb*Ksb';

term_a = Ksa_invVa*Kab*Ksb_invVb';
term_b = Ksb_invVb*Kab'*Ksa_invVa';

bc = 0.25*logdet(pred_var_a) + 0.25*logdet(pred_var_b)...
    -0.5*logdet(2*Kss+pred_var_a+pred_var_b-term_a-term_b); %-0.5*logdet(2*pred_var_a+2*pred_var_b); 
% comment the start of the line above and uncomment the end bit to get the old hellinger distance
    


distance = sqrt(1 - 2*exp(bc));

% example call
% predictants = (1:100)';
% set_A = rand(5,1)*100;
% set_B = rand(6,1)*100;
% K = @(xs,ys) matrify(@(x,y) fcov('sqdexp',{5,1},x,y),xs,ys);
% V = @(xs,ys) K(xs,ys) + eye(size(xs,1))*1^2;
% Kss =  K(predictants,predictants);
% Ksa = K(predictants,set_A);
% Ksb = K(predictants,set_B);
% Ksa_invVa = K(predictants,set_A)/(V(set_A,set_A));
% Ksb_invVb = K(predictants,set_B)/(V(set_B,set_B));
% Kab = K(set_A,set_B);
% hellinger_distance(Kss, Ksa, Ksb, Ksa_invVa, Ksb_invVb, Kab)